from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsEndpointsDirectPredictRequest(_messages.Message):
    """A AiplatformProjectsLocationsEndpointsDirectPredictRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to serve the
      prediction. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1beta1DirectPredictRequest: A
      GoogleCloudAiplatformV1beta1DirectPredictRequest resource to be passed
      as the request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1beta1DirectPredictRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1DirectPredictRequest', 2)