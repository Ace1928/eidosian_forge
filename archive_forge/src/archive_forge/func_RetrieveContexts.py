from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def RetrieveContexts(self, request, global_params=None):
    """Retrieves relevant contexts for a query.

      Args:
        request: (AiplatformProjectsLocationsRetrieveContextsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1RetrieveContextsResponse) The response message.
      """
    config = self.GetMethodConfig('RetrieveContexts')
    return self._RunMethod(config, request, global_params=global_params)