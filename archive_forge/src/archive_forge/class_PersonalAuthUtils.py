from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
class PersonalAuthUtils(object):
    """Util functions for enabling personal auth session."""

    def __init__(self):
        pass

    def _RunOpensslCommand(self, openssl_executable, args, stdin=None):
        """Run the specified command, capturing and returning output as appropriate.

    Args:
      openssl_executable: The path to the openssl executable.
      args: The arguments to the openssl command to run.
      stdin: The input to the command.

    Returns:
      The output of the command.

    Raises:
      PersonalAuthError: If the call to openssl fails
    """
        command = [openssl_executable]
        command.extend(args)
        stderr = None
        try:
            if getattr(subprocess, 'run', None):
                proc = subprocess.run(command, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                stderr = proc.stderr.decode('utf-8').strip()
                proc.check_returncode()
                return proc.stdout
            else:
                p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, _ = p.communicate(input=stdin)
                return stdout
        except Exception as ex:
            if stderr:
                log.error('OpenSSL command "%s" failed with error message "%s"', ' '.join(command), stderr)
            raise exceptions.PersonalAuthError('Failure running openssl command: "' + ' '.join(command) + '": ' + six.text_type(ex))

    def _ComputeHmac(self, key, data, openssl_executable):
        """Compute HMAC tag using OpenSSL."""
        cmd_output = self._RunOpensslCommand(openssl_executable, ['dgst', '-sha256', '-hmac', key], stdin=data).decode('utf-8')
        try:
            stripped_output = cmd_output.strip().split(' ')[1]
            if len(stripped_output) != 64:
                raise ValueError('HMAC output is expected to be 64 characters long.')
            int(stripped_output, 16)
        except Exception as ex:
            raise exceptions.PersonalAuthError('Failure due to invalid openssl output: ' + six.text_type(ex))
        return (stripped_output + '\n').encode('utf-8')

    def _DeriveHkdfKey(self, prk, info, openssl_executable):
        """Derives HMAC-based Key Derivation Function (HKDF) key through expansion on the initial pseudorandom key.

    Args:
      prk: a pseudorandom key.
      info: optional context and application specific information (can be
        empty).
      openssl_executable: The path to the openssl executable.

    Returns:
      Output keying material, expected to be of 256-bit length.
    """
        if len(prk) != 32:
            raise ValueError('The given initial pseudorandom key is expected to be 32 bytes long.')
        base16_prk = base64.b16encode(prk).decode('utf-8')
        t1 = self._ComputeHmac(base16_prk, b'', openssl_executable)
        t2data = bytearray(t1)
        t2data.extend(info)
        t2data.extend(b'\x01')
        return self._ComputeHmac(base16_prk, t2data, openssl_executable)

    @retry.RetryOnException(max_retrials=1)
    def _EncodeTokenUsingOpenssl(self, public_key, secret, openssl_executable):
        """Encode token using OpenSSL.

    Args:
      public_key: The public key for the session/cluster.
      secret: Token to be encrypted.
      openssl_executable: The path to the openssl executable.

    Returns:
      Encrypted token.
    """
        key_hash = hashlib.sha256((public_key + '\n').encode('utf-8')).hexdigest()
        iv_bytes = base64.b16encode(os.urandom(16))
        initialization_vector = iv_bytes.decode('utf-8')
        initial_key = os.urandom(32)
        encryption_key = self._DeriveHkdfKey(initial_key, 'encryption_key'.encode('utf-8'), openssl_executable)
        auth_key = base64.b16encode(self._DeriveHkdfKey(initial_key, 'auth_key'.encode('utf-8'), openssl_executable)).decode('utf-8')
        with tempfile.NamedTemporaryFile() as kf:
            kf.write(public_key.encode('utf-8'))
            kf.seek(0)
            encrypted_key = self._RunOpensslCommand(openssl_executable, ['rsautl', '-oaep', '-encrypt', '-pubin', '-inkey', kf.name], stdin=base64.b64encode(initial_key))
        if len(encrypted_key) != 512:
            raise ValueError('The encrypted key is expected to be 512 bytes long.')
        encoded_key = base64.b64encode(encrypted_key).decode('utf-8')
        with tempfile.NamedTemporaryFile() as pf:
            pf.write(encryption_key)
            pf.seek(0)
            encrypt_args = ['enc', '-aes-256-ctr', '-salt', '-iv', initialization_vector, '-pass', 'file:{}'.format(pf.name)]
            encrypted_token = self._RunOpensslCommand(openssl_executable, encrypt_args, stdin=secret.encode('utf-8'))
        if len(encrypted_key) != 512:
            raise ValueError('The encrypted key is expected to be 512 bytes long.')
        encoded_token = base64.b64encode(encrypted_token).decode('utf-8')
        hmac_input = bytearray(iv_bytes)
        hmac_input.extend(encrypted_token)
        hmac_tag = self._ComputeHmac(auth_key, hmac_input, openssl_executable).decode('utf-8')[0:32]
        return '{}:{}:{}:{}:{}'.format(key_hash, encoded_token, encoded_key, initialization_vector, hmac_tag)

    def EncryptWithPublicKey(self, public_key, secret, openssl_executable):
        """Encrypt secret with resource public key.

    Args:
      public_key: The public key for the session/cluster.
      secret: Token to be encrypted.
      openssl_executable: The path to the openssl executable.

    Returns:
      Encrypted token.
    """
        if openssl_executable:
            return self._EncodeTokenUsingOpenssl(public_key, secret, openssl_executable)
        try:
            import tink
            from tink import hybrid
        except ImportError:
            raise exceptions.PersonalAuthError('Cannot load the Tink cryptography library. Either the library is not installed, or site packages are not enabled for the Google Cloud SDK. Please consult Cloud Dataproc Personal Auth documentation on adding Tink to Google Cloud SDK for further instructions.\nhttps://cloud.google.com/dataproc/docs/concepts/iam/personal-auth')
        hybrid.register()
        context = b''
        public_key_value = json.loads(public_key)['key'][0]['keyData']['value']
        key_hash = hashlib.sha256((public_key_value + '\n').encode('utf-8')).hexdigest()
        reader = tink.JsonKeysetReader(public_key)
        kh_pub = tink.read_no_secret_keyset_handle(reader)
        encrypter = kh_pub.primitive(hybrid.HybridEncrypt)
        ciphertext = encrypter.encrypt(secret.encode('utf-8'), context)
        encoded_token = base64.b64encode(ciphertext).decode('utf-8')
        return '{}:{}'.format(key_hash, encoded_token)

    def IsTinkLibraryInstalled(self):
        """Check if Tink cryptography library can be loaded."""
        try:
            import tink
            from tink import hybrid
            return True
        except ImportError:
            return False