from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class RoleAssignmentsService(base_api.BaseApiService):
    """Service class for the roleAssignments resource."""
    _NAME = u'roleAssignments'

    def __init__(self, client):
        super(AdminDirectoryV1.RoleAssignmentsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a role assignment.

      Args:
        request: (DirectoryRoleAssignmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectoryRoleAssignmentsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.roleAssignments.delete', ordered_params=[u'customer', u'roleAssignmentId'], path_params=[u'customer', u'roleAssignmentId'], query_params=[], relative_path=u'customer/{customer}/roleassignments/{roleAssignmentId}', request_field='', request_type_name=u'DirectoryRoleAssignmentsDeleteRequest', response_type_name=u'DirectoryRoleAssignmentsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve a role assignment.

      Args:
        request: (DirectoryRoleAssignmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (RoleAssignment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.roleAssignments.get', ordered_params=[u'customer', u'roleAssignmentId'], path_params=[u'customer', u'roleAssignmentId'], query_params=[], relative_path=u'customer/{customer}/roleassignments/{roleAssignmentId}', request_field='', request_type_name=u'DirectoryRoleAssignmentsGetRequest', response_type_name=u'RoleAssignment', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a role assignment.

      Args:
        request: (DirectoryRoleAssignmentsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (RoleAssignment) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.roleAssignments.insert', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[], relative_path=u'customer/{customer}/roleassignments', request_field=u'roleAssignment', request_type_name=u'DirectoryRoleAssignmentsInsertRequest', response_type_name=u'RoleAssignment', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a paginated list of all roleAssignments.

      Args:
        request: (DirectoryRoleAssignmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (RoleAssignments) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.roleAssignments.list', ordered_params=[u'customer'], path_params=[u'customer'], query_params=[u'maxResults', u'pageToken', u'roleId', u'userKey'], relative_path=u'customer/{customer}/roleassignments', request_field='', request_type_name=u'DirectoryRoleAssignmentsListRequest', response_type_name=u'RoleAssignments', supports_download=False)