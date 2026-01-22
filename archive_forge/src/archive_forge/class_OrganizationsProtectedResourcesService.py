from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.kmsinventory.v1 import kmsinventory_v1_messages as messages
class OrganizationsProtectedResourcesService(base_api.BaseApiService):
    """Service class for the organizations_protectedResources resource."""
    _NAME = 'organizations_protectedResources'

    def __init__(self, client):
        super(KmsinventoryV1.OrganizationsProtectedResourcesService, self).__init__(client)
        self._upload_configs = {}

    def Search(self, request, global_params=None):
        """Returns metadata about the resources protected by the given Cloud KMS CryptoKey in the given Cloud organization.

      Args:
        request: (KmsinventoryOrganizationsProtectedResourcesSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchProtectedResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/protectedResources:search', http_method='GET', method_id='kmsinventory.organizations.protectedResources.search', ordered_params=['scope'], path_params=['scope'], query_params=['cryptoKey', 'pageSize', 'pageToken', 'resourceTypes'], relative_path='v1/{+scope}/protectedResources:search', request_field='', request_type_name='KmsinventoryOrganizationsProtectedResourcesSearchRequest', response_type_name='SearchProtectedResourcesResponse', supports_download=False)