from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
class ProjectsLocationsEdgeSlmsService(base_api.BaseApiService):
    """Service class for the projects_locations_edgeSlms resource."""
    _NAME = 'projects_locations_edgeSlms'

    def __init__(self, client):
        super(TelcoautomationV1.ProjectsLocationsEdgeSlmsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new EdgeSlm in a given project and location.

      Args:
        request: (TelcoautomationProjectsLocationsEdgeSlmsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeSlms', http_method='POST', method_id='telcoautomation.projects.locations.edgeSlms.create', ordered_params=['parent'], path_params=['parent'], query_params=['edgeSlmId', 'requestId'], relative_path='v1/{+parent}/edgeSlms', request_field='edgeSlm', request_type_name='TelcoautomationProjectsLocationsEdgeSlmsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single EdgeSlm.

      Args:
        request: (TelcoautomationProjectsLocationsEdgeSlmsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeSlms/{edgeSlmsId}', http_method='DELETE', method_id='telcoautomation.projects.locations.edgeSlms.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsEdgeSlmsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single EdgeSlm.

      Args:
        request: (TelcoautomationProjectsLocationsEdgeSlmsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EdgeSlm) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeSlms/{edgeSlmsId}', http_method='GET', method_id='telcoautomation.projects.locations.edgeSlms.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='TelcoautomationProjectsLocationsEdgeSlmsGetRequest', response_type_name='EdgeSlm', supports_download=False)

    def List(self, request, global_params=None):
        """Lists EdgeSlms in a given project and location.

      Args:
        request: (TelcoautomationProjectsLocationsEdgeSlmsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEdgeSlmsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/edgeSlms', http_method='GET', method_id='telcoautomation.projects.locations.edgeSlms.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/edgeSlms', request_field='', request_type_name='TelcoautomationProjectsLocationsEdgeSlmsListRequest', response_type_name='ListEdgeSlmsResponse', supports_download=False)