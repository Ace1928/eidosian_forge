from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securesourcemanager.v1 import securesourcemanager_v1_messages as messages
def IssueRedirectTicketInternal(self, request, global_params=None):
    """THIS METHOD IS FOR INTERNAL USE ONLY.

      Args:
        request: (SecuresourcemanagerProjectsLocationsInstancesIssueRedirectTicketInternalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IssueRedirectTicketInternalResponse) The response message.
      """
    config = self.GetMethodConfig('IssueRedirectTicketInternal')
    return self._RunMethod(config, request, global_params=global_params)