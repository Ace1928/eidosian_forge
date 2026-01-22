from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
def FetchStaticIps(self, request, global_params=None):
    """Fetches a set of static IP addresses that need to be allowlisted by the customer when using the static-IP connectivity method.

      Args:
        request: (DatamigrationProjectsLocationsFetchStaticIpsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchStaticIpsResponse) The response message.
      """
    config = self.GetMethodConfig('FetchStaticIps')
    return self._RunMethod(config, request, global_params=global_params)