from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.alloydb.v1beta import alloydb_v1beta_messages as messages
def InjectFault(self, request, global_params=None):
    """Injects fault in an instance. Imperative only.

      Args:
        request: (AlloydbProjectsLocationsClustersInstancesInjectFaultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('InjectFault')
    return self._RunMethod(config, request, global_params=global_params)