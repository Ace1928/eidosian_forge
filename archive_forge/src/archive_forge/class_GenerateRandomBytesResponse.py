from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateRandomBytesResponse(_messages.Message):
    """Response message for KeyManagementService.GenerateRandomBytes.

  Fields:
    data: The generated data.
    dataCrc32c: Integrity verification field. A CRC32C checksum of the
      returned GenerateRandomBytesResponse.data. An integrity check of
      GenerateRandomBytesResponse.data can be performed by computing the
      CRC32C checksum of GenerateRandomBytesResponse.data and comparing your
      results to this field. Discard the response in case of non-matching
      checksum values, and perform a limited number of retries. A persistent
      mismatch may indicate an issue in your computation of the CRC32C
      checksum. Note: This field is defined as int64 for reasons of
      compatibility across different languages. However, it is a non-negative
      integer, which will never exceed 2^32-1, and can be safely downconverted
      to uint32 in languages that support this type.
  """
    data = _messages.BytesField(1)
    dataCrc32c = _messages.IntegerField(2)