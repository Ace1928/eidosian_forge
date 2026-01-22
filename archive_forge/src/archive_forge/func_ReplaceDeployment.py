from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gsuiteaddons.v1 import gsuiteaddons_v1_messages as messages
def ReplaceDeployment(self, request, global_params=None):
    """Creates or replaces a deployment with the specified name.

      Args:
        request: (GsuiteaddonsProjectsDeploymentsReplaceDeploymentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudGsuiteaddonsV1Deployment) The response message.
      """
    config = self.GetMethodConfig('ReplaceDeployment')
    return self._RunMethod(config, request, global_params=global_params)