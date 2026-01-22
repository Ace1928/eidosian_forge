from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsWorkerPoolsService(base_api.BaseApiService):
    """Service class for the projects_locations_workerPools resource."""
    _NAME = 'projects_locations_workerPools'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsWorkerPoolsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsLocationsWorkerPoolsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools', http_method='POST', method_id='cloudbuild.projects.locations.workerPools.create', ordered_params=['parent'], path_params=['parent'], query_params=['validateOnly', 'workerPoolId'], relative_path='v1/{+parent}/workerPools', request_field='workerPool', request_type_name='CloudbuildProjectsLocationsWorkerPoolsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsLocationsWorkerPoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools/{workerPoolsId}', http_method='DELETE', method_id='cloudbuild.projects.locations.workerPools.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'etag', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsWorkerPoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns details of a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsLocationsWorkerPoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkerPool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools/{workerPoolsId}', http_method='GET', method_id='cloudbuild.projects.locations.workerPools.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsWorkerPoolsGetRequest', response_type_name='WorkerPool', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `WorkerPool`s.

      Args:
        request: (CloudbuildProjectsLocationsWorkerPoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkerPoolsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools', http_method='GET', method_id='cloudbuild.projects.locations.workerPools.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/workerPools', request_field='', request_type_name='CloudbuildProjectsLocationsWorkerPoolsListRequest', response_type_name='ListWorkerPoolsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a `WorkerPool`.

      Args:
        request: (CloudbuildProjectsLocationsWorkerPoolsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/workerPools/{workerPoolsId}', http_method='PATCH', method_id='cloudbuild.projects.locations.workerPools.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask', 'validateOnly'], relative_path='v1/{+name}', request_field='workerPool', request_type_name='CloudbuildProjectsLocationsWorkerPoolsPatchRequest', response_type_name='Operation', supports_download=False)