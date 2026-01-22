from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.alloydb.v1beta import alloydb_v1beta_messages as messages
def GetConnectionInfo(self, request, global_params=None):
    """Get instance metadata used for a connection.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesGetConnectionInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConnectionInfo) The response message.
      """
    config = self.GetMethodConfig('GetConnectionInfo')
    return self._RunMethod(config, request, global_params=global_params)