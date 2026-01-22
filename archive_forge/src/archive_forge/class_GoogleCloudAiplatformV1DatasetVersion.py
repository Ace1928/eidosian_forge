from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1DatasetVersion(_messages.Message):
    """Describes the dataset version.

  Fields:
    bigQueryDatasetName: Output only. Name of the associated BigQuery dataset.
    createTime: Output only. Timestamp when this DatasetVersion was created.
    displayName: The user-defined name of the DatasetVersion. The name can be
      up to 128 characters long and can consist of any UTF-8 characters.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    metadata: Required. Output only. Additional information about the
      DatasetVersion.
    name: Output only. The resource name of the DatasetVersion.
    updateTime: Output only. Timestamp when this DatasetVersion was last
      updated.
  """
    bigQueryDatasetName = _messages.StringField(1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    metadata = _messages.MessageField('extra_types.JsonValue', 5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)