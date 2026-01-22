from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v2 import tpu_v2_messages as messages
class ProjectsLocationsQueuedResourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_queuedResources resource."""
    _NAME = 'projects_locations_queuedResources'

    def __init__(self, client):
        super(TpuV2.ProjectsLocationsQueuedResourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a QueuedResource TPU instance.

      Args:
        request: (TpuProjectsLocationsQueuedResourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queuedResources', http_method='POST', method_id='tpu.projects.locations.queuedResources.create', ordered_params=['parent'], path_params=['parent'], query_params=['queuedResourceId', 'requestId'], relative_path='v2/{+parent}/queuedResources', request_field='queuedResource', request_type_name='TpuProjectsLocationsQueuedResourcesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a QueuedResource TPU instance.

      Args:
        request: (TpuProjectsLocationsQueuedResourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queuedResources/{queuedResourcesId}', http_method='DELETE', method_id='tpu.projects.locations.queuedResources.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v2/{+name}', request_field='', request_type_name='TpuProjectsLocationsQueuedResourcesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a queued resource.

      Args:
        request: (TpuProjectsLocationsQueuedResourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueuedResource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queuedResources/{queuedResourcesId}', http_method='GET', method_id='tpu.projects.locations.queuedResources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='TpuProjectsLocationsQueuedResourcesGetRequest', response_type_name='QueuedResource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists queued resources.

      Args:
        request: (TpuProjectsLocationsQueuedResourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListQueuedResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queuedResources', http_method='GET', method_id='tpu.projects.locations.queuedResources.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/queuedResources', request_field='', request_type_name='TpuProjectsLocationsQueuedResourcesListRequest', response_type_name='ListQueuedResourcesResponse', supports_download=False)

    def Reset(self, request, global_params=None):
        """Resets a QueuedResource TPU instance.

      Args:
        request: (TpuProjectsLocationsQueuedResourcesResetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Reset')
        return self._RunMethod(config, request, global_params=global_params)
    Reset.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/queuedResources/{queuedResourcesId}:reset', http_method='POST', method_id='tpu.projects.locations.queuedResources.reset', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:reset', request_field='resetQueuedResourceRequest', request_type_name='TpuProjectsLocationsQueuedResourcesResetRequest', response_type_name='Operation', supports_download=False)