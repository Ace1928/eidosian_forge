from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvgroupsService(base_api.BaseApiService):
    """Service class for the organizations_envgroups resource."""
    _NAME = 'organizations_envgroups'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvgroupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups', http_method='POST', method_id='apigee.organizations.envgroups.create', ordered_params=['parent'], path_params=['parent'], query_params=['name'], relative_path='v1/{+parent}/envgroups', request_field='googleCloudApigeeV1EnvironmentGroup', request_type_name='ApigeeOrganizationsEnvgroupsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}', http_method='DELETE', method_id='apigee.organizations.envgroups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1EnvironmentGroup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}', http_method='GET', method_id='apigee.organizations.envgroups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsGetRequest', response_type_name='GoogleCloudApigeeV1EnvironmentGroup', supports_download=False)

    def GetDeployedIngressConfig(self, request, global_params=None):
        """Gets the deployed ingress configuration for an environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsGetDeployedIngressConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1EnvironmentGroupConfig) The response message.
      """
        config = self.GetMethodConfig('GetDeployedIngressConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetDeployedIngressConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}/deployedIngressConfig', http_method='GET', method_id='apigee.organizations.envgroups.getDeployedIngressConfig', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsGetDeployedIngressConfigRequest', response_type_name='GoogleCloudApigeeV1EnvironmentGroupConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all environment groups.

      Args:
        request: (ApigeeOrganizationsEnvgroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListEnvironmentGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups', http_method='GET', method_id='apigee.organizations.envgroups.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/envgroups', request_field='', request_type_name='ApigeeOrganizationsEnvgroupsListRequest', response_type_name='GoogleCloudApigeeV1ListEnvironmentGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an environment group.

      Args:
        request: (ApigeeOrganizationsEnvgroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/envgroups/{envgroupsId}', http_method='PATCH', method_id='apigee.organizations.envgroups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1EnvironmentGroup', request_type_name='ApigeeOrganizationsEnvgroupsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)