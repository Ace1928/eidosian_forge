from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def SendCapture(self, request, global_params=None):
    """Send encoded debug capture data for component.

      Args:
        request: (DataflowProjectsLocationsJobsDebugSendCaptureRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SendDebugCaptureResponse) The response message.
      """
    config = self.GetMethodConfig('SendCapture')
    return self._RunMethod(config, request, global_params=global_params)