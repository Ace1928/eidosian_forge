from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetDeployedIngressConfig(self, request, global_params=None):
    """Gets the deployed ingress configuration for an organization.

      Args:
        request: (ApigeeOrganizationsGetDeployedIngressConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1IngressConfig) The response message.
      """
    config = self.GetMethodConfig('GetDeployedIngressConfig')
    return self._RunMethod(config, request, global_params=global_params)