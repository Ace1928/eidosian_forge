from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import crc32c
from googlecloudsdk.command_lib.kms import e2e_integrity
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _CreateDecryptRequest(self, args):
    if args.ciphertext_file == '-' and args.additional_authenticated_data_file == '-':
        raise exceptions.InvalidArgumentException('--ciphertext-file', '--ciphertext-file and --additional-authenticated-data-file cannot both read from stdin.')
    try:
        ciphertext = self._ReadFileOrStdin(args.ciphertext_file, max_bytes=2 * 65536)
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read ciphertext file [{0}]: {1}'.format(args.ciphertext_file, e))
    aad = None
    if args.additional_authenticated_data_file:
        try:
            aad = self._ReadFileOrStdin(args.additional_authenticated_data_file, max_bytes=65536)
        except files.Error as e:
            raise exceptions.BadFileException('Failed to read additional authenticated data file [{0}]: {1}'.format(args.additional_authenticated_data_file, e))
    crypto_key_ref = flags.ParseCryptoKeyName(args)
    if '/cryptoKeyVersions/' in crypto_key_ref.cryptoKeysId:
        raise exceptions.InvalidArgumentException('--key', '{} includes cryptoKeyVersion which is not valid for decrypt.'.format(crypto_key_ref.cryptoKeysId))
    messages = cloudkms_base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysDecryptRequest(name=crypto_key_ref.RelativeName())
    if self._PerformIntegrityVerification(args):
        ciphertext_crc32c = crc32c.Crc32c(ciphertext)
        aad_crc32c = crc32c.Crc32c(aad) if aad is not None else crc32c.Crc32c(b'')
        req.decryptRequest = messages.DecryptRequest(ciphertext=ciphertext, additionalAuthenticatedData=aad, ciphertextCrc32c=ciphertext_crc32c, additionalAuthenticatedDataCrc32c=aad_crc32c)
    else:
        req.decryptRequest = messages.DecryptRequest(ciphertext=ciphertext, additionalAuthenticatedData=aad)
    return req