from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def QueryDeployedModels(self, request, global_params=None):
    """List DeployedModels that have been deployed on this DeploymentResourcePool.

      Args:
        request: (AiplatformProjectsLocationsDeploymentResourcePoolsQueryDeployedModelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1QueryDeployedModelsResponse) The response message.
      """
    config = self.GetMethodConfig('QueryDeployedModels')
    return self._RunMethod(config, request, global_params=global_params)