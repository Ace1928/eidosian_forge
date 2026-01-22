from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPublishersModelsCountTokensRequest(_messages.Message):
    """A AiplatformProjectsLocationsPublishersModelsCountTokensRequest object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to perform token
      counting. Format:
      `projects/{project}/locations/{location}/endpoints/{endpoint}`
    googleCloudAiplatformV1beta1CountTokensRequest: A
      GoogleCloudAiplatformV1beta1CountTokensRequest resource to be passed as
      the request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1beta1CountTokensRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1CountTokensRequest', 2)