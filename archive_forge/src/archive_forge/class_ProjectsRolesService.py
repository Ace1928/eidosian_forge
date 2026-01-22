from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v1 import iam_v1_messages as messages
class ProjectsRolesService(base_api.BaseApiService):
    """Service class for the projects_roles resource."""
    _NAME = 'projects_roles'

    def __init__(self, client):
        super(IamV1.ProjectsRolesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new custom Role.

      Args:
        request: (IamProjectsRolesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Role) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/roles', http_method='POST', method_id='iam.projects.roles.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/roles', request_field='createRoleRequest', request_type_name='IamProjectsRolesCreateRequest', response_type_name='Role', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a custom Role. When you delete a custom role, the following changes occur immediately: * You cannot bind a principal to the custom role in an IAM Policy. * Existing bindings to the custom role are not changed, but they have no effect. * By default, the response from ListRoles does not include the custom role. You have 7 days to undelete the custom role. After 7 days, the following changes occur: * The custom role is permanently deleted and cannot be recovered. * If an IAM policy contains a binding to the custom role, the binding is permanently removed.

      Args:
        request: (IamProjectsRolesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Role) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/roles/{rolesId}', http_method='DELETE', method_id='iam.projects.roles.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsRolesDeleteRequest', response_type_name='Role', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the definition of a Role.

      Args:
        request: (IamProjectsRolesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Role) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/roles/{rolesId}', http_method='GET', method_id='iam.projects.roles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='IamProjectsRolesGetRequest', response_type_name='Role', supports_download=False)

    def List(self, request, global_params=None):
        """Lists every predefined Role that IAM supports, or every custom role that is defined for an organization or project.

      Args:
        request: (IamProjectsRolesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRolesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/roles', http_method='GET', method_id='iam.projects.roles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted', 'view'], relative_path='v1/{+parent}/roles', request_field='', request_type_name='IamProjectsRolesListRequest', response_type_name='ListRolesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the definition of a custom Role.

      Args:
        request: (IamProjectsRolesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Role) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/roles/{rolesId}', http_method='PATCH', method_id='iam.projects.roles.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='role', request_type_name='IamProjectsRolesPatchRequest', response_type_name='Role', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a custom Role.

      Args:
        request: (IamProjectsRolesUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Role) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/roles/{rolesId}:undelete', http_method='POST', method_id='iam.projects.roles.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteRoleRequest', request_type_name='IamProjectsRolesUndeleteRequest', response_type_name='Role', supports_download=False)