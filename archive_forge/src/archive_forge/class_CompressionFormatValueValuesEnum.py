from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompressionFormatValueValuesEnum(_messages.Enum):
    """Optional. The compression type associated with the stored data. If
    unspecified, the data is uncompressed.

    Values:
      COMPRESSION_FORMAT_UNSPECIFIED: CompressionFormat unspecified. Implies
        uncompressed data.
      GZIP: GZip compressed set of files.
      BZIP2: BZip2 compressed set of files.
    """
    COMPRESSION_FORMAT_UNSPECIFIED = 0
    GZIP = 1
    BZIP2 = 2