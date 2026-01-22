from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudresourcemanager.v1 import cloudresourcemanager_v1_messages as messages
def GetAncestry(self, request, global_params=None):
    """Gets a list of ancestors in the resource hierarchy for the Project identified by the specified `project_id` (for example, `my-project-123`). The caller must have read permissions for this Project.

      Args:
        request: (CloudresourcemanagerProjectsGetAncestryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetAncestryResponse) The response message.
      """
    config = self.GetMethodConfig('GetAncestry')
    return self._RunMethod(config, request, global_params=global_params)