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
class AsymmetricDecrypt(base.Command):
    """Decrypt an input file using an asymmetric-encryption key version.

  Decrypts the given ciphertext file using the provided asymmetric-encryption
  key version and saves the decrypted data to the plaintext file.

  By default, the command performs integrity verification on data sent to and
  received from Cloud KMS. Use `--skip-integrity-verification` to disable
  integrity verification.

  ## EXAMPLES
  The following command will read the file '/tmp/my/secret.file.enc', decrypt it
  using the asymmetric CryptoKey `dont-panic` Version 3 and write the plaintext
  to '/tmp/my/secret.file.dec'.

    $ {command} \\
    --location=us-central1 \\
    --keyring=hitchhiker \\
    --key=dont-panic \\
    --version=3 \\
    --ciphertext-file=/tmp/my/secret.file.enc \\
    --plaintext-file=/tmp/my/secret.file.dec

  """

    @staticmethod
    def Args(parser):
        flags.AddKeyResourceFlags(parser, 'to use for asymmetric-decryption.')
        flags.AddCryptoKeyVersionFlag(parser, 'to use for asymmetric-decryption')
        flags.AddCiphertextFileFlag(parser, 'to decrypt')
        flags.AddPlaintextFileFlag(parser, 'to output')
        flags.AddSkipIntegrityVerification(parser)

    def _PerformIntegrityVerification(self, args):
        return not args.skip_integrity_verification

    def _CreateAsymmetricDecryptRequest(self, args):
        try:
            ciphertext = console_io.ReadFromFileOrStdin(args.ciphertext_file, binary=True)
        except files.Error as e:
            raise exceptions.BadFileException('Failed to read ciphertext file [{0}]: {1}'.format(args.ciphertext_file, e))
        messages = cloudkms_base.GetMessagesModule()
        crypto_key_ref = flags.ParseCryptoKeyVersionName(args)
        req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsAsymmetricDecryptRequest(name=crypto_key_ref.RelativeName())
        if self._PerformIntegrityVerification(args):
            ciphertext_crc32c = crc32c.Crc32c(ciphertext)
            req.asymmetricDecryptRequest = messages.AsymmetricDecryptRequest(ciphertext=ciphertext, ciphertextCrc32c=ciphertext_crc32c)
        else:
            req.asymmetricDecryptRequest = messages.AsymmetricDecryptRequest(ciphertext=ciphertext)
        return req

    def _VerifyResponseIntegrityFields(self, req, resp):
        """Verifies integrity fields in response."""
        if not resp.verifiedCiphertextCrc32c:
            raise e2e_integrity.ClientSideIntegrityVerificationError(e2e_integrity.GetRequestToServerCorruptedErrorMessage())
        if not crc32c.Crc32cMatches(resp.plaintext, resp.plaintextCrc32c):
            raise e2e_integrity.ClientSideIntegrityVerificationError(e2e_integrity.GetResponseFromServerCorruptedErrorMessage())

    def Run(self, args):
        req = self._CreateAsymmetricDecryptRequest(args)
        client = cloudkms_base.GetClientInstance()
        try:
            resp = client.projects_locations_keyRings_cryptoKeys_cryptoKeyVersions.AsymmetricDecrypt(req)
        except apitools_exceptions.HttpBadRequestError as error:
            e2e_integrity.ProcessHttpBadRequestError(error)
        if self._PerformIntegrityVerification(args):
            self._VerifyResponseIntegrityFields(req, resp)
        try:
            log.WriteToFileOrStdout(args.plaintext_file, resp.plaintext or '', overwrite=True, binary=True, private=True)
        except files.Error as e:
            raise exceptions.BadFileException(e)