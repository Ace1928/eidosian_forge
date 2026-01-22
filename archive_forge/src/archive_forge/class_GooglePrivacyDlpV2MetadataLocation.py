from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2MetadataLocation(_messages.Message):
    """Metadata Location

  Enums:
    TypeValueValuesEnum: Type of metadata containing the finding.

  Fields:
    storageLabel: Storage metadata.
    type: Type of metadata containing the finding.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of metadata containing the finding.

    Values:
      METADATATYPE_UNSPECIFIED: Unused
      STORAGE_METADATA: General file metadata provided by Cloud Storage.
    """
        METADATATYPE_UNSPECIFIED = 0
        STORAGE_METADATA = 1
    storageLabel = _messages.MessageField('GooglePrivacyDlpV2StorageMetadataLabel', 1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)