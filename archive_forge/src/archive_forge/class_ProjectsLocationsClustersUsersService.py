from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.alloydb.v1beta import alloydb_v1beta_messages as messages
class ProjectsLocationsClustersUsersService(base_api.BaseApiService):
    """Service class for the projects_locations_clusters_users resource."""
    _NAME = 'projects_locations_clusters_users'

    def __init__(self, client):
        super(AlloydbV1beta.ProjectsLocationsClustersUsersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new User in a given project, location, and cluster.

      Args:
        request: (AlloydbProjectsLocationsClustersUsersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (User) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/users', http_method='POST', method_id='alloydb.projects.locations.clusters.users.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'userId', 'validateOnly'], relative_path='v1beta/{+parent}/users', request_field='user', request_type_name='AlloydbProjectsLocationsClustersUsersCreateRequest', response_type_name='User', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single User.

      Args:
        request: (AlloydbProjectsLocationsClustersUsersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/users/{usersId}', http_method='DELETE', method_id='alloydb.projects.locations.clusters.users.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'validateOnly'], relative_path='v1beta/{+name}', request_field='', request_type_name='AlloydbProjectsLocationsClustersUsersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single User.

      Args:
        request: (AlloydbProjectsLocationsClustersUsersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (User) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/users/{usersId}', http_method='GET', method_id='alloydb.projects.locations.clusters.users.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AlloydbProjectsLocationsClustersUsersGetRequest', response_type_name='User', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Users in a given project and location.

      Args:
        request: (AlloydbProjectsLocationsClustersUsersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUsersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/users', http_method='GET', method_id='alloydb.projects.locations.clusters.users.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/users', request_field='', request_type_name='AlloydbProjectsLocationsClustersUsersListRequest', response_type_name='ListUsersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single User.

      Args:
        request: (AlloydbProjectsLocationsClustersUsersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (User) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/clusters/{clustersId}/users/{usersId}', http_method='PATCH', method_id='alloydb.projects.locations.clusters.users.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1beta/{+name}', request_field='user', request_type_name='AlloydbProjectsLocationsClustersUsersPatchRequest', response_type_name='User', supports_download=False)