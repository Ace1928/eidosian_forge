from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.assuredworkloads.v1 import assuredworkloads_v1_messages as messages
def AnalyzeWorkloadMove(self, request, global_params=None):
    """Analyzes a hypothetical move of a source resource to a target workload to surface compliance risks. The analysis is best effort and is not guaranteed to be exhaustive.

      Args:
        request: (AssuredworkloadsOrganizationsLocationsWorkloadsAnalyzeWorkloadMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAssuredworkloadsV1AnalyzeWorkloadMoveResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeWorkloadMove')
    return self._RunMethod(config, request, global_params=global_params)