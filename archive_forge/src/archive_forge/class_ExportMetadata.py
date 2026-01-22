from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportMetadata(_messages.Message):
    """ExportMetadata represents the metadata of the exported artifacts. The
  metadata.json file in export artifact can be parsed as this message

  Enums:
    SourceValueValuesEnum: The source type of the migration.

  Fields:
    exportEncryptionKey: Encryption key that was used to encrypt the export
      artifacts.
    filePaths: List of files created as part of export artifact (excluding the
      metadata). The paths are relative to the folder containing the metadata.
    lookerEncryptionKey: Looker encryption key, encrypted with the provided
      export encryption key. This value will only be populated if the looker
      instance uses Looker managed encryption instead of CMEK.
    lookerInstance: Name of the exported instance. Format:
      projects/{project}/locations/{location}/instances/{instance}
    lookerPlatformEdition: Platform edition of the exported instance.
    lookerVersion: Version of instance when the export was created.
    source: The source type of the migration.
  """

    class SourceValueValuesEnum(_messages.Enum):
        """The source type of the migration.

    Values:
      SOURCE_UNSPECIFIED: Source not specified
      LOOKER_CORE: Source of export is Looker Core
      LOOKER_ORIGINAL: Source of export is Looker Original
    """
        SOURCE_UNSPECIFIED = 0
        LOOKER_CORE = 1
        LOOKER_ORIGINAL = 2
    exportEncryptionKey = _messages.MessageField('ExportMetadataEncryptionKey', 1)
    filePaths = _messages.StringField(2, repeated=True)
    lookerEncryptionKey = _messages.StringField(3)
    lookerInstance = _messages.StringField(4)
    lookerPlatformEdition = _messages.StringField(5)
    lookerVersion = _messages.StringField(6)
    source = _messages.EnumField('SourceValueValuesEnum', 7)