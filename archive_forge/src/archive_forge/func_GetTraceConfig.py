from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetTraceConfig(self, request, global_params=None):
    """Get distributed trace configuration in an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetTraceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfig) The response message.
      """
    config = self.GetMethodConfig('GetTraceConfig')
    return self._RunMethod(config, request, global_params=global_params)