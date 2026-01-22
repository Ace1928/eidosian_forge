from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ContentLocation(_messages.Message):
    """Precise location of the finding within a document, record, image, or
  metadata container.

  Fields:
    containerName: Name of the container where the finding is located. The top
      level name is the source file name or table name. Names of some common
      storage containers are formatted as follows: * BigQuery tables:
      `{project_id}:{dataset_id}.{table_id}` * Cloud Storage files:
      `gs://{bucket}/{path}` * Datastore namespace: {namespace} Nested names
      could be absent if the embedded object has no string identifier (for
      example, an image contained within a document).
    containerTimestamp: Finding container modification timestamp, if
      applicable. For Cloud Storage, this field contains the last file
      modification timestamp. For a BigQuery table, this field contains the
      last_modified_time property. For Datastore, this field isn't populated.
    containerVersion: Finding container version, if available ("generation"
      for Cloud Storage).
    documentLocation: Location data for document files.
    imageLocation: Location within an image's pixels.
    metadataLocation: Location within the metadata for inspected content.
    recordLocation: Location within a row or record of a database table.
  """
    containerName = _messages.StringField(1)
    containerTimestamp = _messages.StringField(2)
    containerVersion = _messages.StringField(3)
    documentLocation = _messages.MessageField('GooglePrivacyDlpV2DocumentLocation', 4)
    imageLocation = _messages.MessageField('GooglePrivacyDlpV2ImageLocation', 5)
    metadataLocation = _messages.MessageField('GooglePrivacyDlpV2MetadataLocation', 6)
    recordLocation = _messages.MessageField('GooglePrivacyDlpV2RecordLocation', 7)