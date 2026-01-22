from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsNamespacesRbacrolebindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_namespaces_rbacrolebindings resource."""
    _NAME = 'projects_locations_namespaces_rbacrolebindings'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsNamespacesRbacrolebindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsNamespacesRbacrolebindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/rbacrolebindings', http_method='POST', method_id='gkehub.projects.locations.namespaces.rbacrolebindings.create', ordered_params=['parent'], path_params=['parent'], query_params=['rbacrolebindingId'], relative_path='v1beta/{+parent}/rbacrolebindings', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsNamespacesRbacrolebindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsNamespacesRbacrolebindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/rbacrolebindings/{rbacrolebindingsId}', http_method='DELETE', method_id='gkehub.projects.locations.namespaces.rbacrolebindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsNamespacesRbacrolebindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsNamespacesRbacrolebindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RBACRoleBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/rbacrolebindings/{rbacrolebindingsId}', http_method='GET', method_id='gkehub.projects.locations.namespaces.rbacrolebindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsNamespacesRbacrolebindingsGetRequest', response_type_name='RBACRoleBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsNamespacesRbacrolebindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRBACRoleBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/rbacrolebindings', http_method='GET', method_id='gkehub.projects.locations.namespaces.rbacrolebindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/rbacrolebindings', request_field='', request_type_name='GkehubProjectsLocationsNamespacesRbacrolebindingsListRequest', response_type_name='ListRBACRoleBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsNamespacesRbacrolebindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/namespaces/{namespacesId}/rbacrolebindings/{rbacrolebindingsId}', http_method='PATCH', method_id='gkehub.projects.locations.namespaces.rbacrolebindings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsNamespacesRbacrolebindingsPatchRequest', response_type_name='Operation', supports_download=False)