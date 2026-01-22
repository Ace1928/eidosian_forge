from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesImportFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesImportFeatureValues
  Request object.

  Fields:
    entityType: Required. The resource name of the EntityType grouping the
      Features for which values are being imported. Format: `projects/{project
      }/locations/{location}/featurestores/{featurestore}/entityTypes/{entityT
      ype}`
    googleCloudAiplatformV1ImportFeatureValuesRequest: A
      GoogleCloudAiplatformV1ImportFeatureValuesRequest resource to be passed
      as the request body.
  """
    entityType = _messages.StringField(1, required=True)
    googleCloudAiplatformV1ImportFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1ImportFeatureValuesRequest', 2)