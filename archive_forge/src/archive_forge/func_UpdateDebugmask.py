from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateDebugmask(self, request, global_params=None):
    """Updates the debug mask singleton resource for an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUpdateDebugmaskRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugMask) The response message.
      """
    config = self.GetMethodConfig('UpdateDebugmask')
    return self._RunMethod(config, request, global_params=global_params)