from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsResourceValueConfigsService(base_api.BaseApiService):
    """Service class for the organizations_resourceValueConfigs resource."""
    _NAME = 'organizations_resourceValueConfigs'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsResourceValueConfigsService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates a ResourceValueConfig for an organization. Maps user's tags to difference resource values for use by the attack path simulation.

      Args:
        request: (SecuritycenterOrganizationsResourceValueConfigsBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchCreateResourceValueConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/resourceValueConfigs:batchCreate', http_method='POST', method_id='securitycenter.organizations.resourceValueConfigs.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/resourceValueConfigs:batchCreate', request_field='batchCreateResourceValueConfigsRequest', request_type_name='SecuritycenterOrganizationsResourceValueConfigsBatchCreateRequest', response_type_name='BatchCreateResourceValueConfigsResponse', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a ResourceValueConfig.

      Args:
        request: (SecuritycenterOrganizationsResourceValueConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/resourceValueConfigs/{resourceValueConfigsId}', http_method='DELETE', method_id='securitycenter.organizations.resourceValueConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsResourceValueConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a ResourceValueConfig.

      Args:
        request: (SecuritycenterOrganizationsResourceValueConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2ResourceValueConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/resourceValueConfigs/{resourceValueConfigsId}', http_method='GET', method_id='securitycenter.organizations.resourceValueConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsResourceValueConfigsGetRequest', response_type_name='GoogleCloudSecuritycenterV2ResourceValueConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all ResourceValueConfigs.

      Args:
        request: (SecuritycenterOrganizationsResourceValueConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourceValueConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/resourceValueConfigs', http_method='GET', method_id='securitycenter.organizations.resourceValueConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/resourceValueConfigs', request_field='', request_type_name='SecuritycenterOrganizationsResourceValueConfigsListRequest', response_type_name='ListResourceValueConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing ResourceValueConfigs with new rules.

      Args:
        request: (SecuritycenterOrganizationsResourceValueConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2ResourceValueConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/resourceValueConfigs/{resourceValueConfigsId}', http_method='PATCH', method_id='securitycenter.organizations.resourceValueConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2ResourceValueConfig', request_type_name='SecuritycenterOrganizationsResourceValueConfigsPatchRequest', response_type_name='GoogleCloudSecuritycenterV2ResourceValueConfig', supports_download=False)