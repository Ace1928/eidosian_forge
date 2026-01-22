from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPublishersModelsComputeTokensRequest(_messages.Message):
    """A AiplatformProjectsLocationsPublishersModelsComputeTokensRequest
  object.

  Fields:
    endpoint: Required. The name of the Endpoint requested to get lists of
      tokens and token ids.
    googleCloudAiplatformV1beta1ComputeTokensRequest: A
      GoogleCloudAiplatformV1beta1ComputeTokensRequest resource to be passed
      as the request body.
  """
    endpoint = _messages.StringField(1, required=True)
    googleCloudAiplatformV1beta1ComputeTokensRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1ComputeTokensRequest', 2)