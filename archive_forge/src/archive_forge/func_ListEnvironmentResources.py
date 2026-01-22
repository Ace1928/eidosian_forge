from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def ListEnvironmentResources(self, request, global_params=None):
    """Lists all resource files, optionally filtering by type. For more information about resource files, see [Resource files](https://cloud.google.com/apigee/docs/api-platform/develop/resource-files).

      Args:
        request: (ApigeeOrganizationsEnvironmentsResourcefilesListEnvironmentResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListEnvironmentResourcesResponse) The response message.
      """
    config = self.GetMethodConfig('ListEnvironmentResources')
    return self._RunMethod(config, request, global_params=global_params)