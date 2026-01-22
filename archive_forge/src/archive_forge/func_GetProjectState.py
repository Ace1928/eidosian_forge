from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def GetProjectState(self, request, global_params=None):
    """Gets state of a single `Project`.

      Args:
        request: (VmwareengineProjectsLocationsGetProjectStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectState) The response message.
      """
    config = self.GetMethodConfig('GetProjectState')
    return self._RunMethod(config, request, global_params=global_params)