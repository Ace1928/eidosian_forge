from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.assuredworkloads.v1 import assuredworkloads_v1_messages as messages
def EnableResourceMonitoring(self, request, global_params=None):
    """Enable resource violation monitoring for a workload.

      Args:
        request: (AssuredworkloadsOrganizationsLocationsWorkloadsEnableResourceMonitoringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAssuredworkloadsV1EnableResourceMonitoringResponse) The response message.
      """
    config = self.GetMethodConfig('EnableResourceMonitoring')
    return self._RunMethod(config, request, global_params=global_params)