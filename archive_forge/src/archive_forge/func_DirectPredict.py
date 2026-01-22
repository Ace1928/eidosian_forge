from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def DirectPredict(self, request, global_params=None):
    """Perform an unary online prediction request to a gRPC model server for Vertex first-party products and frameworks.

      Args:
        request: (AiplatformProjectsLocationsEndpointsDirectPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1DirectPredictResponse) The response message.
      """
    config = self.GetMethodConfig('DirectPredict')
    return self._RunMethod(config, request, global_params=global_params)