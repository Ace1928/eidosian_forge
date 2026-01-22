from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
class OrganizationsGcpUserAccessBindingsService(base_api.BaseApiService):
    """Service class for the organizations_gcpUserAccessBindings resource."""
    _NAME = 'organizations_gcpUserAccessBindings'

    def __init__(self, client):
        super(AccesscontextmanagerV1alpha.OrganizationsGcpUserAccessBindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a GcpUserAccessBinding. If the client specifies a name, the server ignores it. Fails if a resource already exists with the same group_key. Completion of this long-running operation does not necessarily signify that the new binding is deployed onto all affected users, which may take more time.

      Args:
        request: (AccesscontextmanagerOrganizationsGcpUserAccessBindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/gcpUserAccessBindings', http_method='POST', method_id='accesscontextmanager.organizations.gcpUserAccessBindings.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/gcpUserAccessBindings', request_field='gcpUserAccessBinding', request_type_name='AccesscontextmanagerOrganizationsGcpUserAccessBindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a GcpUserAccessBinding. Completion of this long-running operation does not necessarily signify that the binding deletion is deployed onto all affected users, which may take more time.

      Args:
        request: (AccesscontextmanagerOrganizationsGcpUserAccessBindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/gcpUserAccessBindings/{gcpUserAccessBindingsId}', http_method='DELETE', method_id='accesscontextmanager.organizations.gcpUserAccessBindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerOrganizationsGcpUserAccessBindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the GcpUserAccessBinding with the given name.

      Args:
        request: (AccesscontextmanagerOrganizationsGcpUserAccessBindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GcpUserAccessBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/gcpUserAccessBindings/{gcpUserAccessBindingsId}', http_method='GET', method_id='accesscontextmanager.organizations.gcpUserAccessBindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerOrganizationsGcpUserAccessBindingsGetRequest', response_type_name='GcpUserAccessBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all GcpUserAccessBindings for a Google Cloud organization.

      Args:
        request: (AccesscontextmanagerOrganizationsGcpUserAccessBindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGcpUserAccessBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/gcpUserAccessBindings', http_method='GET', method_id='accesscontextmanager.organizations.gcpUserAccessBindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/gcpUserAccessBindings', request_field='', request_type_name='AccesscontextmanagerOrganizationsGcpUserAccessBindingsListRequest', response_type_name='ListGcpUserAccessBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a GcpUserAccessBinding. Completion of this long-running operation does not necessarily signify that the changed binding is deployed onto all affected users, which may take more time.

      Args:
        request: (AccesscontextmanagerOrganizationsGcpUserAccessBindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/gcpUserAccessBindings/{gcpUserAccessBindingsId}', http_method='PATCH', method_id='accesscontextmanager.organizations.gcpUserAccessBindings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='gcpUserAccessBinding', request_type_name='AccesscontextmanagerOrganizationsGcpUserAccessBindingsPatchRequest', response_type_name='Operation', supports_download=False)