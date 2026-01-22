from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.assuredworkloads.v1 import assuredworkloads_v1_messages as messages
def MutatePartnerPermissions(self, request, global_params=None):
    """Update the permissions settings for an existing partner workload. For force updates don't set etag field in the Workload. Only one update operation per workload can be in progress.

      Args:
        request: (AssuredworkloadsOrganizationsLocationsWorkloadsMutatePartnerPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAssuredworkloadsV1Workload) The response message.
      """
    config = self.GetMethodConfig('MutatePartnerPermissions')
    return self._RunMethod(config, request, global_params=global_params)