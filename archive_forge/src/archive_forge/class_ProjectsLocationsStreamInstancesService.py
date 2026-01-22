from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.stream.v1 import stream_v1_messages as messages
class ProjectsLocationsStreamInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_streamInstances resource."""
    _NAME = 'projects_locations_streamInstances'

    def __init__(self, client):
        super(StreamV1.ProjectsLocationsStreamInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new StreamInstance that manages the turnup and rollout of the streaming service for a given StreamContent. The returned Operation can be used to track the creation status by polling operations.get. The Operation will complete when the creation is done. Returns [StreamInstance] in the Operation.response field on successful completion.

      Args:
        request: (StreamProjectsLocationsStreamInstancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamInstances', http_method='POST', method_id='stream.projects.locations.streamInstances.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'streamInstanceId'], relative_path='v1/{+parent}/streamInstances', request_field='streamInstance', request_type_name='StreamProjectsLocationsStreamInstancesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single StreamInstance. This method tears down the streaming service of the associated StreamContent. The returned Operation can be used to track the deletion status by polling operations.get. The Operation will complete when the deletion is done. Returns Empty in the Operation.response field on successful completion.

      Args:
        request: (StreamProjectsLocationsStreamInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamInstances/{streamInstancesId}', http_method='DELETE', method_id='stream.projects.locations.streamInstances.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='StreamProjectsLocationsStreamInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single StreamInstance.

      Args:
        request: (StreamProjectsLocationsStreamInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StreamInstance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamInstances/{streamInstancesId}', http_method='GET', method_id='stream.projects.locations.streamInstances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StreamProjectsLocationsStreamInstancesGetRequest', response_type_name='StreamInstance', supports_download=False)

    def List(self, request, global_params=None):
        """Lists StreamInstances in a given project and location.

      Args:
        request: (StreamProjectsLocationsStreamInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListStreamInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamInstances', http_method='GET', method_id='stream.projects.locations.streamInstances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/streamInstances', request_field='', request_type_name='StreamProjectsLocationsStreamInstancesListRequest', response_type_name='ListStreamInstancesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single StreamInstance.

      Args:
        request: (StreamProjectsLocationsStreamInstancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/streamInstances/{streamInstancesId}', http_method='PATCH', method_id='stream.projects.locations.streamInstances.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='streamInstance', request_type_name='StreamProjectsLocationsStreamInstancesPatchRequest', response_type_name='Operation', supports_download=False)