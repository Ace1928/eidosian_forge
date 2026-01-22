from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def UpdateTraceConfig(self, request, global_params=None):
    """Updates the trace configurations in an environment. Note that the repeated fields have replace semantics when included in the field mask and that they will be overwritten by the value of the fields in the request body.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUpdateTraceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfig) The response message.
      """
    config = self.GetMethodConfig('UpdateTraceConfig')
    return self._RunMethod(config, request, global_params=global_params)