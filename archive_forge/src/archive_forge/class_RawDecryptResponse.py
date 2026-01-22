from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RawDecryptResponse(_messages.Message):
    """Response message for KeyManagementService.RawDecrypt.

  Enums:
    ProtectionLevelValueValuesEnum: The ProtectionLevel of the
      CryptoKeyVersion used in decryption.

  Fields:
    plaintext: The decrypted data.
    plaintextCrc32c: Integrity verification field. A CRC32C checksum of the
      returned RawDecryptResponse.plaintext. An integrity check of plaintext
      can be performed by computing the CRC32C checksum of plaintext and
      comparing your results to this field. Discard the response in case of
      non-matching checksum values, and perform a limited number of retries. A
      persistent mismatch may indicate an issue in your computation of the
      CRC32C checksum. Note: receiving this response message indicates that
      KeyManagementService is able to successfully decrypt the ciphertext.
      Note: This field is defined as int64 for reasons of compatibility across
      different languages. However, it is a non-negative integer, which will
      never exceed 2^32-1, and can be safely downconverted to uint32 in
      languages that support this type.
    protectionLevel: The ProtectionLevel of the CryptoKeyVersion used in
      decryption.
    verifiedAdditionalAuthenticatedDataCrc32c: Integrity verification field. A
      flag indicating whether
      RawDecryptRequest.additional_authenticated_data_crc32c was received by
      KeyManagementService and used for the integrity verification of
      additional_authenticated_data. A false value of this field indicates
      either that // RawDecryptRequest.additional_authenticated_data_crc32c
      was left unset or that it was not delivered to KeyManagementService. If
      you've set RawDecryptRequest.additional_authenticated_data_crc32c but
      this field is still false, discard the response and perform a limited
      number of retries.
    verifiedCiphertextCrc32c: Integrity verification field. A flag indicating
      whether RawDecryptRequest.ciphertext_crc32c was received by
      KeyManagementService and used for the integrity verification of the
      ciphertext. A false value of this field indicates either that
      RawDecryptRequest.ciphertext_crc32c was left unset or that it was not
      delivered to KeyManagementService. If you've set
      RawDecryptRequest.ciphertext_crc32c but this field is still false,
      discard the response and perform a limited number of retries.
    verifiedInitializationVectorCrc32c: Integrity verification field. A flag
      indicating whether RawDecryptRequest.initialization_vector_crc32c was
      received by KeyManagementService and used for the integrity verification
      of initialization_vector. A false value of this field indicates either
      that RawDecryptRequest.initialization_vector_crc32c was left unset or
      that it was not delivered to KeyManagementService. If you've set
      RawDecryptRequest.initialization_vector_crc32c but this field is still
      false, discard the response and perform a limited number of retries.
  """

    class ProtectionLevelValueValuesEnum(_messages.Enum):
        """The ProtectionLevel of the CryptoKeyVersion used in decryption.

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
    plaintext = _messages.BytesField(1)
    plaintextCrc32c = _messages.IntegerField(2)
    protectionLevel = _messages.EnumField('ProtectionLevelValueValuesEnum', 3)
    verifiedAdditionalAuthenticatedDataCrc32c = _messages.BooleanField(4)
    verifiedCiphertextCrc32c = _messages.BooleanField(5)
    verifiedInitializationVectorCrc32c = _messages.BooleanField(6)