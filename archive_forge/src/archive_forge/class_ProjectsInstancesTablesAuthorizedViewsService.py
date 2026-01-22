from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class ProjectsInstancesTablesAuthorizedViewsService(base_api.BaseApiService):
    """Service class for the projects_instances_tables_authorizedViews resource."""
    _NAME = 'projects_instances_tables_authorizedViews'

    def __init__(self, client):
        super(BigtableadminV2.ProjectsInstancesTablesAuthorizedViewsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AuthorizedView in a table.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews', http_method='POST', method_id='bigtableadmin.projects.instances.tables.authorizedViews.create', ordered_params=['parent'], path_params=['parent'], query_params=['authorizedViewId'], relative_path='v2/{+parent}/authorizedViews', request_field='authorizedView', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Permanently deletes a specified AuthorizedView.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews/{authorizedViewsId}', http_method='DELETE', method_id='bigtableadmin.projects.instances.tables.authorizedViews.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information from a specified AuthorizedView.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizedView) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews/{authorizedViewsId}', http_method='GET', method_id='bigtableadmin.projects.instances.tables.authorizedViews.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsGetRequest', response_type_name='AuthorizedView', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a Bigtable resource. Returns an empty policy if the resource exists but does not have a policy set.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews/{authorizedViewsId}:getIamPolicy', http_method='POST', method_id='bigtableadmin.projects.instances.tables.authorizedViews.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all AuthorizedViews from a specific table.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizedViewsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews', http_method='GET', method_id='bigtableadmin.projects.instances.tables.authorizedViews.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v2/{+parent}/authorizedViews', request_field='', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsListRequest', response_type_name='ListAuthorizedViewsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an AuthorizedView in a table.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews/{authorizedViewsId}', http_method='PATCH', method_id='bigtableadmin.projects.instances.tables.authorizedViews.patch', ordered_params=['name'], path_params=['name'], query_params=['ignoreWarnings', 'updateMask'], relative_path='v2/{+name}', request_field='authorizedView', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a Bigtable resource. Replaces any existing policy.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews/{authorizedViewsId}:setIamPolicy', http_method='POST', method_id='bigtableadmin.projects.instances.tables.authorizedViews.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that the caller has on the specified Bigtable resource.

      Args:
        request: (BigtableadminProjectsInstancesTablesAuthorizedViewsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}/authorizedViews/{authorizedViewsId}:testIamPermissions', http_method='POST', method_id='bigtableadmin.projects.instances.tables.authorizedViews.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BigtableadminProjectsInstancesTablesAuthorizedViewsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)