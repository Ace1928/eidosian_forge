from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
def BulkMute(self, request, global_params=None):
    """Kicks off an LRO to bulk mute findings for a parent based on a filter. If no location is specified, findings are muted in global. The parent can be either an organization, folder, or project. The findings matched by the filter will be muted after the LRO is done.

      Args:
        request: (SecuritycenterProjectsLocationsFindingsBulkMuteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('BulkMute')
    return self._RunMethod(config, request, global_params=global_params)