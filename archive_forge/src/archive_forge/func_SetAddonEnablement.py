from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def SetAddonEnablement(self, request, global_params=None):
    """Updates an add-on enablement status of an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsAddonsConfigSetAddonEnablementRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('SetAddonEnablement')
    return self._RunMethod(config, request, global_params=global_params)