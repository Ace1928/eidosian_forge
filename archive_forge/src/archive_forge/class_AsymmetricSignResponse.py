from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AsymmetricSignResponse(_messages.Message):
    """Response message for KeyManagementService.AsymmetricSign.

  Enums:
    ProtectionLevelValueValuesEnum: The ProtectionLevel of the
      CryptoKeyVersion used for signing.

  Fields:
    name: The resource name of the CryptoKeyVersion used for signing. Check
      this field to verify that the intended resource was used for signing.
    protectionLevel: The ProtectionLevel of the CryptoKeyVersion used for
      signing.
    signature: The created signature.
    signatureCrc32c: Integrity verification field. A CRC32C checksum of the
      returned AsymmetricSignResponse.signature. An integrity check of
      AsymmetricSignResponse.signature can be performed by computing the
      CRC32C checksum of AsymmetricSignResponse.signature and comparing your
      results to this field. Discard the response in case of non-matching
      checksum values, and perform a limited number of retries. A persistent
      mismatch may indicate an issue in your computation of the CRC32C
      checksum. Note: This field is defined as int64 for reasons of
      compatibility across different languages. However, it is a non-negative
      integer, which will never exceed 2^32-1, and can be safely downconverted
      to uint32 in languages that support this type.
    verifiedDataCrc32c: Integrity verification field. A flag indicating
      whether AsymmetricSignRequest.data_crc32c was received by
      KeyManagementService and used for the integrity verification of the
      data. A false value of this field indicates either that
      AsymmetricSignRequest.data_crc32c was left unset or that it was not
      delivered to KeyManagementService. If you've set
      AsymmetricSignRequest.data_crc32c but this field is still false, discard
      the response and perform a limited number of retries.
    verifiedDigestCrc32c: Integrity verification field. A flag indicating
      whether AsymmetricSignRequest.digest_crc32c was received by
      KeyManagementService and used for the integrity verification of the
      digest. A false value of this field indicates either that
      AsymmetricSignRequest.digest_crc32c was left unset or that it was not
      delivered to KeyManagementService. If you've set
      AsymmetricSignRequest.digest_crc32c but this field is still false,
      discard the response and perform a limited number of retries.
  """

    class ProtectionLevelValueValuesEnum(_messages.Enum):
        """The ProtectionLevel of the CryptoKeyVersion used for signing.

    Values:
      PROTECTION_LEVEL_UNSPECIFIED: Not specified.
      SOFTWARE: Crypto operations are performed in software.
      HSM: Crypto operations are performed in a Hardware Security Module.
      EXTERNAL: Crypto operations are performed by an external key manager.
      EXTERNAL_VPC: Crypto operations are performed in an EKM-over-VPC
        backend.
    """
        PROTECTION_LEVEL_UNSPECIFIED = 0
        SOFTWARE = 1
        HSM = 2
        EXTERNAL = 3
        EXTERNAL_VPC = 4
    name = _messages.StringField(1)
    protectionLevel = _messages.EnumField('ProtectionLevelValueValuesEnum', 2)
    signature = _messages.BytesField(3)
    signatureCrc32c = _messages.IntegerField(4)
    verifiedDataCrc32c = _messages.BooleanField(5)
    verifiedDigestCrc32c = _messages.BooleanField(6)