from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ReadUsage(self, request, global_params=None):
    """Returns a list of monthly active users for a given TensorBoard instance.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsReadUsageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardUsageResponse) The response message.
      """
    config = self.GetMethodConfig('ReadUsage')
    return self._RunMethod(config, request, global_params=global_params)