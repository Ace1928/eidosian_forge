from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsGroupsService(base_api.BaseApiService):
    """Service class for the projects_locations_groups resource."""
    _NAME = 'projects_locations_groups'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsGroupsService, self).__init__(client)
        self._upload_configs = {}

    def AddGroupMigration(self, request, global_params=None):
        """Adds a MigratingVm to a Group.

      Args:
        request: (VmmigrationProjectsLocationsGroupsAddGroupMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddGroupMigration')
        return self._RunMethod(config, request, global_params=global_params)
    AddGroupMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups/{groupsId}:addGroupMigration', http_method='POST', method_id='vmmigration.projects.locations.groups.addGroupMigration', ordered_params=['group'], path_params=['group'], query_params=[], relative_path='v1/{+group}:addGroupMigration', request_field='addGroupMigrationRequest', request_type_name='VmmigrationProjectsLocationsGroupsAddGroupMigrationRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Group in a given project and location.

      Args:
        request: (VmmigrationProjectsLocationsGroupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups', http_method='POST', method_id='vmmigration.projects.locations.groups.create', ordered_params=['parent'], path_params=['parent'], query_params=['groupId', 'requestId'], relative_path='v1/{+parent}/groups', request_field='group', request_type_name='VmmigrationProjectsLocationsGroupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Group.

      Args:
        request: (VmmigrationProjectsLocationsGroupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups/{groupsId}', http_method='DELETE', method_id='vmmigration.projects.locations.groups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsGroupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Group.

      Args:
        request: (VmmigrationProjectsLocationsGroupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Group) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups/{groupsId}', http_method='GET', method_id='vmmigration.projects.locations.groups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsGroupsGetRequest', response_type_name='Group', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Groups in a given project and location.

      Args:
        request: (VmmigrationProjectsLocationsGroupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGroupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups', http_method='GET', method_id='vmmigration.projects.locations.groups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/groups', request_field='', request_type_name='VmmigrationProjectsLocationsGroupsListRequest', response_type_name='ListGroupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Group.

      Args:
        request: (VmmigrationProjectsLocationsGroupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups/{groupsId}', http_method='PATCH', method_id='vmmigration.projects.locations.groups.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='group', request_type_name='VmmigrationProjectsLocationsGroupsPatchRequest', response_type_name='Operation', supports_download=False)

    def RemoveGroupMigration(self, request, global_params=None):
        """Removes a MigratingVm from a Group.

      Args:
        request: (VmmigrationProjectsLocationsGroupsRemoveGroupMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveGroupMigration')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveGroupMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/groups/{groupsId}:removeGroupMigration', http_method='POST', method_id='vmmigration.projects.locations.groups.removeGroupMigration', ordered_params=['group'], path_params=['group'], query_params=[], relative_path='v1/{+group}:removeGroupMigration', request_field='removeGroupMigrationRequest', request_type_name='VmmigrationProjectsLocationsGroupsRemoveGroupMigrationRequest', response_type_name='Operation', supports_download=False)