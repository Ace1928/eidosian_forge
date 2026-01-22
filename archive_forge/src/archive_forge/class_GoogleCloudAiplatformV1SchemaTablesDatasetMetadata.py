from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTablesDatasetMetadata(_messages.Message):
    """The metadata of Datasets that contain tables data.

  Fields:
    inputConfig: A
      GoogleCloudAiplatformV1SchemaTablesDatasetMetadataInputConfig attribute.
  """
    inputConfig = _messages.MessageField('GoogleCloudAiplatformV1SchemaTablesDatasetMetadataInputConfig', 1)