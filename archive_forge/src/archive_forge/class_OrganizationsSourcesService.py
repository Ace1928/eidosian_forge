from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsSourcesService(base_api.BaseApiService):
    """Service class for the organizations_sources resource."""
    _NAME = 'organizations_sources'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsSourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a source.

      Args:
        request: (SecuritycenterOrganizationsSourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Source) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources', http_method='POST', method_id='securitycenter.organizations.sources.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/sources', request_field='source', request_type_name='SecuritycenterOrganizationsSourcesCreateRequest', response_type_name='Source', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a source.

      Args:
        request: (SecuritycenterOrganizationsSourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Source) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources/{sourcesId}', http_method='GET', method_id='securitycenter.organizations.sources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsSourcesGetRequest', response_type_name='Source', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy on the specified Source.

      Args:
        request: (SecuritycenterOrganizationsSourcesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources/{sourcesId}:getIamPolicy', http_method='POST', method_id='securitycenter.organizations.sources.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='SecuritycenterOrganizationsSourcesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all sources belonging to an organization.

      Args:
        request: (SecuritycenterOrganizationsSourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources', http_method='GET', method_id='securitycenter.organizations.sources.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/sources', request_field='', request_type_name='SecuritycenterOrganizationsSourcesListRequest', response_type_name='ListSourcesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a source.

      Args:
        request: (SecuritycenterOrganizationsSourcesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Source) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources/{sourcesId}', http_method='PATCH', method_id='securitycenter.organizations.sources.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='source', request_type_name='SecuritycenterOrganizationsSourcesPatchRequest', response_type_name='Source', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified Source.

      Args:
        request: (SecuritycenterOrganizationsSourcesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources/{sourcesId}:setIamPolicy', http_method='POST', method_id='securitycenter.organizations.sources.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='SecuritycenterOrganizationsSourcesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the permissions that a caller has on the specified source.

      Args:
        request: (SecuritycenterOrganizationsSourcesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/sources/{sourcesId}:testIamPermissions', http_method='POST', method_id='securitycenter.organizations.sources.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SecuritycenterOrganizationsSourcesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)