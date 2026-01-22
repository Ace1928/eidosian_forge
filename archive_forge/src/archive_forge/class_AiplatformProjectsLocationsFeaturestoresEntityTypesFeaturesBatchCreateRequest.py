from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesBatchCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesBatchCreate
  Request object.

  Fields:
    googleCloudAiplatformV1BatchCreateFeaturesRequest: A
      GoogleCloudAiplatformV1BatchCreateFeaturesRequest resource to be passed
      as the request body.
    parent: Required. The resource name of the EntityType to create the batch
      of Features under. Format: `projects/{project}/locations/{location}/feat
      urestores/{featurestore}/entityTypes/{entity_type}`
  """
    googleCloudAiplatformV1BatchCreateFeaturesRequest = _messages.MessageField('GoogleCloudAiplatformV1BatchCreateFeaturesRequest', 1)
    parent = _messages.StringField(2, required=True)