from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _CkmRsaAesKeyWrap(self, import_method, public_key_bytes, target_key_bytes, client, messages):
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import keywrap
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        log.err.Print('Cannot load the Pyca cryptography library. Either the library is not installed, or site packages are not enabled for the Google Cloud SDK. Please consult https://cloud.google.com/kms/docs/crypto for further instructions.')
        sys.exit(1)
    sha = hashes.SHA1()
    if self._IsSha2ImportMethod(import_method, messages):
        sha = hashes.SHA256()
    if not self._IsRsaAesWrappingImportMethod(import_method, messages):
        if import_method == messages.ImportJob.ImportMethodValueValuesEnum.RSA_OAEP_3072_SHA256:
            modulus_byte_length = 3072 // 8
        elif import_method == messages.ImportJob.ImportMethodValueValuesEnum.RSA_OAEP_4096_SHA256:
            modulus_byte_length = 4096 // 8
        else:
            raise ValueError('unexpected import method: {0}'.format(import_method))
        max_target_key_size = modulus_byte_length - 2 * sha.digest_size - 2
        if len(target_key_bytes) > max_target_key_size:
            raise exceptions.BadFileException('target-key-file', "The file is larger than the import method's maximum size of {0} bytes.".format(max_target_key_size))
    aes_wrapped_key = b''
    to_be_rsa_wrapped_key = target_key_bytes
    public_key = serialization.load_pem_public_key(public_key_bytes, backend=default_backend())
    if self._IsRsaAesWrappingImportMethod(import_method, messages):
        to_be_rsa_wrapped_key = os.urandom(32)
        aes_wrapped_key = keywrap.aes_key_wrap_with_padding(to_be_rsa_wrapped_key, target_key_bytes, default_backend())
    rsa_wrapped_key = public_key.encrypt(to_be_rsa_wrapped_key, padding.OAEP(mgf=padding.MGF1(sha), algorithm=sha, label=None))
    return rsa_wrapped_key + aes_wrapped_key