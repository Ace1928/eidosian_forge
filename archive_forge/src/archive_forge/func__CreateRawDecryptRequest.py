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
def _CreateRawDecryptRequest(self, args):
    if args.ciphertext_file == '-' and args.initialization_vector_file == '-':
        raise exceptions.InvalidArgumentException('--ciphertext-file and --initialization-vector-file', "both parameters can't be read from stdin.")
    if args.ciphertext_file == '-' and args.additional_authenticated_data_file == '-':
        raise exceptions.InvalidArgumentException('--ciphertext-file and --additional-authenticated-data-file', "both parameters can't be read from stdin.")
    if args.initialization_vector_file == '-' and args.additional_authenticated_data_file == '-':
        raise exceptions.InvalidArgumentException('--initialization-vector-file and --additional-authenticated-data-file', "both parameters can't be read from stdin.")
    try:
        ciphertext = self._ReadFileOrStdin(args.ciphertext_file, max_bytes=2 * 65536)
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read ciphertext file [{0}]: {1}'.format(args.ciphertext_file, e))
    try:
        iv = self._ReadFileOrStdin(args.initialization_vector_file, max_bytes=CBC_CTR_IV_SIZE)
    except files.Error as e:
        raise exceptions.BadFileException('Failed to read initialization vector file [{0}]: {1}'.format(args.initialization_vector_file, e))
    if len(iv) != CBC_CTR_IV_SIZE:
        raise exceptions.BadFileException('--initialization-vector-file', 'the IV size must be {0} bytes.'.format(CBC_CTR_IV_SIZE))
    aad = b''
    if args.additional_authenticated_data_file:
        try:
            aad = self._ReadFileOrStdin(args.additional_authenticated_data_file, max_bytes=65536)
        except files.Error as e:
            raise exceptions.BadFileException('Failed to read additional authenticated data file [{0}]: {1}'.format(args.additional_authenticated_data_file, e))
    crypto_key_ref = flags.ParseCryptoKeyVersionName(args)
    messages = cloudkms_base.GetMessagesModule()
    request = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawDecryptRequest(name=crypto_key_ref.RelativeName())
    if self._PerformIntegrityVerification(args):
        ciphertext_crc32c = crc32c.Crc32c(ciphertext)
        iv_crc32c = crc32c.Crc32c(iv)
        aad_crc32c = crc32c.Crc32c(aad)
        request.rawDecryptRequest = messages.RawDecryptRequest(ciphertext=ciphertext, initializationVector=iv, additionalAuthenticatedData=aad, ciphertextCrc32c=ciphertext_crc32c, initializationVectorCrc32c=iv_crc32c, additionalAuthenticatedDataCrc32c=aad_crc32c)
    else:
        request.rawDecryptRequest = messages.RawDecryptRequest(ciphertext=ciphertext, initializationVector=iv, additionalAuthenticatedData=aad)
    return request