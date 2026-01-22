from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
class ProjectsLocationsScopesRbacrolebindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_scopes_rbacrolebindings resource."""
    _NAME = 'projects_locations_scopes_rbacrolebindings'

    def __init__(self, client):
        super(GkehubV1beta.ProjectsLocationsScopesRbacrolebindingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Scope RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsScopesRbacrolebindingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/rbacrolebindings', http_method='POST', method_id='gkehub.projects.locations.scopes.rbacrolebindings.create', ordered_params=['parent'], path_params=['parent'], query_params=['rbacrolebindingId'], relative_path='v1beta/{+parent}/rbacrolebindings', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsScopesRbacrolebindingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Scope RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsScopesRbacrolebindingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/rbacrolebindings/{rbacrolebindingsId}', http_method='DELETE', method_id='gkehub.projects.locations.scopes.rbacrolebindings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesRbacrolebindingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details of a Scope RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsScopesRbacrolebindingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RBACRoleBinding) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/rbacrolebindings/{rbacrolebindingsId}', http_method='GET', method_id='gkehub.projects.locations.scopes.rbacrolebindings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='GkehubProjectsLocationsScopesRbacrolebindingsGetRequest', response_type_name='RBACRoleBinding', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all Scope RBACRoleBindings.

      Args:
        request: (GkehubProjectsLocationsScopesRbacrolebindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScopeRBACRoleBindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/rbacrolebindings', http_method='GET', method_id='gkehub.projects.locations.scopes.rbacrolebindings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/rbacrolebindings', request_field='', request_type_name='GkehubProjectsLocationsScopesRbacrolebindingsListRequest', response_type_name='ListScopeRBACRoleBindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Scope RBACRoleBinding.

      Args:
        request: (GkehubProjectsLocationsScopesRbacrolebindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/scopes/{scopesId}/rbacrolebindings/{rbacrolebindingsId}', http_method='PATCH', method_id='gkehub.projects.locations.scopes.rbacrolebindings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='rBACRoleBinding', request_type_name='GkehubProjectsLocationsScopesRbacrolebindingsPatchRequest', response_type_name='Operation', supports_download=False)