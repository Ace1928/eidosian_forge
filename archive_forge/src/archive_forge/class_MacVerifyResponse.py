from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MacVerifyResponse(_messages.Message):
    """Response message for KeyManagementService.MacVerify.

  Enums:
    ProtectionLevelValueValuesEnum: The ProtectionLevel of the
      CryptoKeyVersion used for verification.

  Fields:
    name: The resource name of the CryptoKeyVersion used for verification.
      Check this field to verify that the intended resource was used for
      verification.
    protectionLevel: The ProtectionLevel of the CryptoKeyVersion used for
      verification.
    success: This field indicates whether or not the verification operation
      for MacVerifyRequest.mac over MacVerifyRequest.data was successful.
    verifiedDataCrc32c: Integrity verification field. A flag indicating
      whether MacVerifyRequest.data_crc32c was received by
      KeyManagementService and used for the integrity verification of the
      data. A false value of this field indicates either that
      MacVerifyRequest.data_crc32c was left unset or that it was not delivered
      to KeyManagementService. If you've set MacVerifyRequest.data_crc32c but
      this field is still false, discard the response and perform a limited
      number of retries.
    verifiedMacCrc32c: Integrity verification field. A flag indicating whether
      MacVerifyRequest.mac_crc32c was received by KeyManagementService and
      used for the integrity verification of the data. A false value of this
      field indicates either that MacVerifyRequest.mac_crc32c was left unset
      or that it was not delivered to KeyManagementService. If you've set
      MacVerifyRequest.mac_crc32c but this field is still false, discard the
      response and perform a limited number of retries.
    verifiedSuccessIntegrity: Integrity verification field. This value is used
      for the integrity verification of [MacVerifyResponse.success]. If the
      value of this field contradicts the value of
      [MacVerifyResponse.success], discard the response and perform a limited
      number of retries.
  """

    class ProtectionLevelValueValuesEnum(_messages.Enum):
        """The ProtectionLevel of the CryptoKeyVersion used for verification.

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
    success = _messages.BooleanField(3)
    verifiedDataCrc32c = _messages.BooleanField(4)
    verifiedMacCrc32c = _messages.BooleanField(5)
    verifiedSuccessIntegrity = _messages.BooleanField(6)