from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def ProvisionDeployment(self, request, global_params=None):
    """Creates a new SAS deployment through the GCP workflow. Creates a SAS organization if an organization match is not found.

      Args:
        request: (SasPortalProvisionDeploymentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalProvisionDeploymentResponse) The response message.
      """
    config = self.GetMethodConfig('ProvisionDeployment')
    return self._RunMethod(config, request, global_params=global_params)