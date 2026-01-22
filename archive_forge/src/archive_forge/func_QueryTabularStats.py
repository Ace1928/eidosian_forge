from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def QueryTabularStats(self, request, global_params=None):
    """Retrieve security statistics as tabular rows.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityStatsQueryTabularStatsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1QueryTabularStatsResponse) The response message.
      """
    config = self.GetMethodConfig('QueryTabularStats')
    return self._RunMethod(config, request, global_params=global_params)