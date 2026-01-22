from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def ProvisionOrganization(self, request, global_params=None):
    """Provisions a new Apigee organization with a functioning runtime. This is the standard way to create trial organizations for a free Apigee trial.

      Args:
        request: (ApigeeProjectsProvisionOrganizationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('ProvisionOrganization')
    return self._RunMethod(config, request, global_params=global_params)