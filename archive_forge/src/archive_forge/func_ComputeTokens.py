from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def ComputeTokens(self, request, global_params=None):
    """Return a list of tokens based on the input text.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsComputeTokensRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ComputeTokensResponse) The response message.
      """
    config = self.GetMethodConfig('ComputeTokens')
    return self._RunMethod(config, request, global_params=global_params)