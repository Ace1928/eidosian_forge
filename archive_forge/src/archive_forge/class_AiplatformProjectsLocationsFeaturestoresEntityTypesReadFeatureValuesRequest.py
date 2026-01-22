from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesReadFeatureValuesRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsFeaturestoresEntityTypesReadFeatureValuesRequest
  object.

  Fields:
    entityType: Required. The resource name of the EntityType for the entity
      being read. Value format: `projects/{project}/locations/{location}/featu
      restores/{featurestore}/entityTypes/{entityType}`. For example, for a
      machine learning model predicting user clicks on a website, an
      EntityType ID could be `user`.
    googleCloudAiplatformV1ReadFeatureValuesRequest: A
      GoogleCloudAiplatformV1ReadFeatureValuesRequest resource to be passed as
      the request body.
  """
    entityType = _messages.StringField(1, required=True)
    googleCloudAiplatformV1ReadFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1ReadFeatureValuesRequest', 2)