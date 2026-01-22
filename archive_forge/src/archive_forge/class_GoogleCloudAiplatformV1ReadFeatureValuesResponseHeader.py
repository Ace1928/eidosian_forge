from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ReadFeatureValuesResponseHeader(_messages.Message):
    """Response header with metadata for the requested
  ReadFeatureValuesRequest.entity_type and Features.

  Fields:
    entityType: The resource name of the EntityType from the
      ReadFeatureValuesRequest. Value format: `projects/{project}/locations/{l
      ocation}/featurestores/{featurestore}/entityTypes/{entityType}`.
    featureDescriptors: List of Feature metadata corresponding to each piece
      of ReadFeatureValuesResponse.EntityView.data.
  """
    entityType = _messages.StringField(1)
    featureDescriptors = _messages.MessageField('GoogleCloudAiplatformV1ReadFeatureValuesResponseFeatureDescriptor', 2, repeated=True)