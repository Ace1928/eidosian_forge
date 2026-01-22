from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def ModifyEnvironment(self, request, global_params=None):
    """Updates properties for an Apigee environment with patch semantics using a field mask. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsEnvironmentsModifyEnvironmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('ModifyEnvironment')
    return self._RunMethod(config, request, global_params=global_params)