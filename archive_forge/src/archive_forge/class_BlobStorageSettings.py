from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlobStorageSettings(_messages.Message):
    """Settings for data stored in Blob storage.

  Enums:
    BlobStorageClassValueValuesEnum: The Storage class in which the Blob data
      is stored.

  Fields:
    blobStorageClass: The Storage class in which the Blob data is stored.
  """

    class BlobStorageClassValueValuesEnum(_messages.Enum):
        """The Storage class in which the Blob data is stored.

    Values:
      BLOB_STORAGE_CLASS_UNSPECIFIED: If unspecified in CreateDataset, the
        StorageClass defaults to STANDARD. If unspecified in UpdateDataset and
        the StorageClass is set in the field mask, an InvalidRequest error is
        thrown.
      STANDARD: This stores the Object in Blob Standard Storage:
        https://cloud.google.com/storage/docs/storage-classes#standard
      NEARLINE: This stores the Object in Blob Nearline Storage:
        https://cloud.google.com/storage/docs/storage-classes#nearline
      COLDLINE: This stores the Object in Blob Coldline Storage:
        https://cloud.google.com/storage/docs/storage-classes#coldline
      ARCHIVE: This stores the Object in Blob Archive Storage:
        https://cloud.google.com/storage/docs/storage-classes#archive
    """
        BLOB_STORAGE_CLASS_UNSPECIFIED = 0
        STANDARD = 1
        NEARLINE = 2
        COLDLINE = 3
        ARCHIVE = 4
    blobStorageClass = _messages.EnumField('BlobStorageClassValueValuesEnum', 1)