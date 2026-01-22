from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.v2 import deploymentmanager_v2_messages as messages
def CancelPreview(self, request, global_params=None):
    """Cancels and removes the preview currently associated with the deployment.

      Args:
        request: (DeploymentmanagerDeploymentsCancelPreviewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('CancelPreview')
    return self._RunMethod(config, request, global_params=global_params)