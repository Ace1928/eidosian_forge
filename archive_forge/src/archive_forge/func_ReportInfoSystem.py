from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.notebooks.v2 import notebooks_v2_messages as messages
def ReportInfoSystem(self, request, global_params=None):
    """Allows notebook instances to report their latest instance information to the Notebooks API server. The server will merge the reported information to the instance metadata store. Do not use this method directly.

      Args:
        request: (NotebooksProjectsLocationsInstancesReportInfoSystemRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ReportInfoSystem')
    return self._RunMethod(config, request, global_params=global_params)