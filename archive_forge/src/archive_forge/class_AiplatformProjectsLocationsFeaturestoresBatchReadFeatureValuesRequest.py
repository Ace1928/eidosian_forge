from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresBatchReadFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresBatchReadFeatureValuesRequest
  object.

  Fields:
    featurestore: Required. The resource name of the Featurestore from which
      to query Feature values. Format:
      `projects/{project}/locations/{location}/featurestores/{featurestore}`
    googleCloudAiplatformV1BatchReadFeatureValuesRequest: A
      GoogleCloudAiplatformV1BatchReadFeatureValuesRequest resource to be
      passed as the request body.
  """
    featurestore = _messages.StringField(1, required=True)
    googleCloudAiplatformV1BatchReadFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1BatchReadFeatureValuesRequest', 2)