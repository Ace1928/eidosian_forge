from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaImageDatasetMetadata(_messages.Message):
    """The metadata of Datasets that contain Image DataItems.

  Fields:
    dataItemSchemaUri: Points to a YAML file stored on Google Cloud Storage
      describing payload of the Image DataItems that belong to this Dataset.
    gcsBucket: Google Cloud Storage Bucket name that contains the blob data of
      this Dataset.
  """
    dataItemSchemaUri = _messages.StringField(1)
    gcsBucket = _messages.StringField(2)