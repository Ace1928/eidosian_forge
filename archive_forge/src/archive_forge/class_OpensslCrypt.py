from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
class OpensslCrypt(object):
    """Base Class for OpenSSL encryption functions."""

    def __init__(self, openssl_executable):
        self.openssl_executable = openssl_executable

    def RunOpenSSL(self, cmd_args, cmd_input=None):
        """Run an openssl command with optional input and return the output."""
        command = [self.openssl_executable]
        command.extend(cmd_args)
        try:
            p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, stderr = p.communicate(cmd_input)
            log.debug('Ran command "{0}" with standard error of:\n{1}'.format(' '.join(command), stderr))
        except OSError as e:
            raise OpenSSLException('[{0}] exited with [{1}].'.format(command[0], e.strerror))
        if p.returncode:
            raise OpenSSLException('[{0}] exited with return code [{1}]:\n{2}.'.format(command[0], p.returncode, stderr))
        return output

    def GetKeyPair(self, key_length=DEFAULT_KEY_LENGTH):
        """Returns an RSA key pair (private key)."""
        return self.RunOpenSSL(['genrsa', six.text_type(key_length)])

    def GetPublicKey(self, key):
        """Returns a public key from a key pair."""
        return self.RunOpenSSL(['rsa', '-pubout'], cmd_input=key)

    def DecryptMessage(self, key, enc_message, destroy_key=False):
        """Returns a decrypted message using the provided key.

    Args:
      key: An openssl key pair (private key).
      enc_message: a base64 encoded encrypted message.
      destroy_key: Unused for OpenSSL.
    Returns:
      Decrypted version of enc_message
    """
        del destroy_key
        encrypted_message_data = base64.b64decode(enc_message)
        with tempfile.NamedTemporaryFile() as tf:
            tf.write(key)
            tf.flush()
            openssl_args = ['rsautl', '-decrypt', '-oaep', '-inkey', tf.name]
            message = self.RunOpenSSL(openssl_args, cmd_input=encrypted_message_data)
        return message

    def GetModulusExponentFromPublicKey(self, public_key, key_length=DEFAULT_KEY_LENGTH):
        """Returns a base64 encoded modulus and exponent from the public key."""
        key = StripKey(public_key)
        decoded_key = base64.b64decode(key)
        exponent = decoded_key[-3:]
        key_bytes = key_length // 8
        if key_length % 8:
            key_bytes += 1
        modulus_start = -5 - key_bytes
        modulus = decoded_key[modulus_start:-5]
        b64_mod = base64.b64encode(modulus)
        b64_exp = base64.b64encode(exponent)
        return (b64_mod, b64_exp)