from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MacVerifyRequest(_messages.Message):
    """Request message for KeyManagementService.MacVerify.

  Fields:
    data: Required. The data used previously as a MacSignRequest.data to
      generate the MAC tag.
    dataCrc32c: Optional. An optional CRC32C checksum of the
      MacVerifyRequest.data. If specified, KeyManagementService will verify
      the integrity of the received MacVerifyRequest.data using this checksum.
      KeyManagementService will report an error if the checksum verification
      fails. If you receive a checksum error, your client should verify that
      CRC32C(MacVerifyRequest.data) is equal to MacVerifyRequest.data_crc32c,
      and if so, perform a limited number of retries. A persistent mismatch
      may indicate an issue in your computation of the CRC32C checksum. Note:
      This field is defined as int64 for reasons of compatibility across
      different languages. However, it is a non-negative integer, which will
      never exceed 2^32-1, and can be safely downconverted to uint32 in
      languages that support this type.
    mac: Required. The signature to verify.
    macCrc32c: Optional. An optional CRC32C checksum of the
      MacVerifyRequest.mac. If specified, KeyManagementService will verify the
      integrity of the received MacVerifyRequest.mac using this checksum.
      KeyManagementService will report an error if the checksum verification
      fails. If you receive a checksum error, your client should verify that
      CRC32C(MacVerifyRequest.tag) is equal to MacVerifyRequest.mac_crc32c,
      and if so, perform a limited number of retries. A persistent mismatch
      may indicate an issue in your computation of the CRC32C checksum. Note:
      This field is defined as int64 for reasons of compatibility across
      different languages. However, it is a non-negative integer, which will
      never exceed 2^32-1, and can be safely downconverted to uint32 in
      languages that support this type.
  """
    data = _messages.BytesField(1)
    dataCrc32c = _messages.IntegerField(2)
    mac = _messages.BytesField(3)
    macCrc32c = _messages.IntegerField(4)