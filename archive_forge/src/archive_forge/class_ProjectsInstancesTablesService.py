from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class ProjectsInstancesTablesService(base_api.BaseApiService):
    """Service class for the projects_instances_tables resource."""
    _NAME = 'projects_instances_tables'

    def __init__(self, client):
        super(BigtableadminV2.ProjectsInstancesTablesService, self).__init__(client)
        self._upload_configs = {}

    def CheckConsistency(self, request, global_params=None):
        """Checks replication consistency based on a consistency token, that is, if replication has caught up based on the conditions specified in the token and the check request.

      Args:
        request: (BigtableadminProjectsInstancesTablesCheckConsistencyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckConsistencyResponse) The response message.
      """
        config = self.GetMethodConfig('CheckConsistency')
        return self._RunMethod(config, request, global_params=global_params)
    CheckConsistency.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:checkConsistency', http_method='POST', method_id='bigtableadmin.projects.instances.tables.checkConsistency', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:checkConsistency', request_field='checkConsistencyRequest', request_type_name='BigtableadminProjectsInstancesTablesCheckConsistencyRequest', response_type_name='CheckConsistencyResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new table in the specified instance. The table can be created with a full set of initial column families, specified in the request.

      Args:
        request: (BigtableadminProjectsInstancesTablesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables', http_method='POST', method_id='bigtableadmin.projects.instances.tables.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/tables', request_field='createTableRequest', request_type_name='BigtableadminProjectsInstancesTablesCreateRequest', response_type_name='Table', supports_download=False)

    def Delete(self, request, global_params=None):
        """Permanently deletes a specified table and all of its data.

      Args:
        request: (BigtableadminProjectsInstancesTablesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}', http_method='DELETE', method_id='bigtableadmin.projects.instances.tables.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesTablesDeleteRequest', response_type_name='Empty', supports_download=False)

    def DropRowRange(self, request, global_params=None):
        """Permanently drop/delete a row range from a specified table. The request can specify whether to delete all rows in a table, or only those that match a particular prefix. Note that row key prefixes used here are treated as service data. For more information about how service data is handled, see the [Google Cloud Privacy Notice](https://cloud.google.com/terms/cloud-privacy-notice).

      Args:
        request: (BigtableadminProjectsInstancesTablesDropRowRangeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('DropRowRange')
        return self._RunMethod(config, request, global_params=global_params)
    DropRowRange.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:dropRowRange', http_method='POST', method_id='bigtableadmin.projects.instances.tables.dropRowRange', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:dropRowRange', request_field='dropRowRangeRequest', request_type_name='BigtableadminProjectsInstancesTablesDropRowRangeRequest', response_type_name='Empty', supports_download=False)

    def GenerateConsistencyToken(self, request, global_params=None):
        """Generates a consistency token for a Table, which can be used in CheckConsistency to check whether mutations to the table that finished before this call started have been replicated. The tokens will be available for 90 days.

      Args:
        request: (BigtableadminProjectsInstancesTablesGenerateConsistencyTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateConsistencyTokenResponse) The response message.
      """
        config = self.GetMethodConfig('GenerateConsistencyToken')
        return self._RunMethod(config, request, global_params=global_params)
    GenerateConsistencyToken.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:generateConsistencyToken', http_method='POST', method_id='bigtableadmin.projects.instances.tables.generateConsistencyToken', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:generateConsistencyToken', request_field='generateConsistencyTokenRequest', request_type_name='BigtableadminProjectsInstancesTablesGenerateConsistencyTokenRequest', response_type_name='GenerateConsistencyTokenResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets metadata information about the specified table.

      Args:
        request: (BigtableadminProjectsInstancesTablesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}', http_method='GET', method_id='bigtableadmin.projects.instances.tables.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesTablesGetRequest', response_type_name='Table', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a Bigtable resource. Returns an empty policy if the resource exists but does not have a policy set.

      Args:
        request: (BigtableadminProjectsInstancesTablesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:getIamPolicy', http_method='POST', method_id='bigtableadmin.projects.instances.tables.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='BigtableadminProjectsInstancesTablesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all tables served from a specified instance.

      Args:
        request: (BigtableadminProjectsInstancesTablesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTablesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables', http_method='GET', method_id='bigtableadmin.projects.instances.tables.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v2/{+parent}/tables', request_field='', request_type_name='BigtableadminProjectsInstancesTablesListRequest', response_type_name='ListTablesResponse', supports_download=False)

    def ModifyColumnFamilies(self, request, global_params=None):
        """Performs a series of column family modifications on the specified table. Either all or none of the modifications will occur before this method returns, but data requests received prior to that point may see a table where only some modifications have taken effect.

      Args:
        request: (BigtableadminProjectsInstancesTablesModifyColumnFamiliesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('ModifyColumnFamilies')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyColumnFamilies.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:modifyColumnFamilies', http_method='POST', method_id='bigtableadmin.projects.instances.tables.modifyColumnFamilies', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:modifyColumnFamilies', request_field='modifyColumnFamiliesRequest', request_type_name='BigtableadminProjectsInstancesTablesModifyColumnFamiliesRequest', response_type_name='Table', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a specified table.

      Args:
        request: (BigtableadminProjectsInstancesTablesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}', http_method='PATCH', method_id='bigtableadmin.projects.instances.tables.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='table', request_type_name='BigtableadminProjectsInstancesTablesPatchRequest', response_type_name='Operation', supports_download=False)

    def Restore(self, request, global_params=None):
        """Create a new table by restoring from a completed backup. The returned table long-running operation can be used to track the progress of the operation, and to cancel it. The metadata field type is RestoreTableMetadata. The response type is Table, if successful.

      Args:
        request: (BigtableadminProjectsInstancesTablesRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables:restore', http_method='POST', method_id='bigtableadmin.projects.instances.tables.restore', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/tables:restore', request_field='restoreTableRequest', request_type_name='BigtableadminProjectsInstancesTablesRestoreRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a Bigtable resource. Replaces any existing policy.

      Args:
        request: (BigtableadminProjectsInstancesTablesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:setIamPolicy', http_method='POST', method_id='bigtableadmin.projects.instances.tables.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='BigtableadminProjectsInstancesTablesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that the caller has on the specified Bigtable resource.

      Args:
        request: (BigtableadminProjectsInstancesTablesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:testIamPermissions', http_method='POST', method_id='bigtableadmin.projects.instances.tables.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BigtableadminProjectsInstancesTablesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Restores a specified table which was accidentally deleted.

      Args:
        request: (BigtableadminProjectsInstancesTablesUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/tables/{tablesId}:undelete', http_method='POST', method_id='bigtableadmin.projects.instances.tables.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undelete', request_field='undeleteTableRequest', request_type_name='BigtableadminProjectsInstancesTablesUndeleteRequest', response_type_name='Operation', supports_download=False)