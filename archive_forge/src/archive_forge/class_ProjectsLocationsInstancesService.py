from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.parallelstore.v1beta import parallelstore_v1beta_messages as messages
class ProjectsLocationsInstancesService(base_api.BaseApiService):
    """Service class for the projects_locations_instances resource."""
    _NAME = 'projects_locations_instances'

    def __init__(self, client):
        super(ParallelstoreV1beta.ProjectsLocationsInstancesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Parallelstore instance in a given project and location.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances', http_method='POST', method_id='parallelstore.projects.locations.instances.create', ordered_params=['parent'], path_params=['parent'], query_params=['instanceId', 'requestId'], relative_path='v1beta/{+parent}/instances', request_field='instance', request_type_name='ParallelstoreProjectsLocationsInstancesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Instance.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}', http_method='DELETE', method_id='parallelstore.projects.locations.instances.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='ParallelstoreProjectsLocationsInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def ExportData(self, request, global_params=None):
        """ExportData copies data from Parallelstore to Cloud Storage.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesExportDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ExportData')
        return self._RunMethod(config, request, global_params=global_params)
    ExportData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}:exportData', http_method='POST', method_id='parallelstore.projects.locations.instances.exportData', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:exportData', request_field='exportDataRequest', request_type_name='ParallelstoreProjectsLocationsInstancesExportDataRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Instance.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Instance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}', http_method='GET', method_id='parallelstore.projects.locations.instances.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='ParallelstoreProjectsLocationsInstancesGetRequest', response_type_name='Instance', supports_download=False)

    def ImportData(self, request, global_params=None):
        """ImportData copies data from Cloud Storage to Parallelstore.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesImportDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ImportData')
        return self._RunMethod(config, request, global_params=global_params)
    ImportData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}:importData', http_method='POST', method_id='parallelstore.projects.locations.instances.importData', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}:importData', request_field='importDataRequest', request_type_name='ParallelstoreProjectsLocationsInstancesImportDataRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Instances in a given project and location.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstancesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances', http_method='GET', method_id='parallelstore.projects.locations.instances.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/instances', request_field='', request_type_name='ParallelstoreProjectsLocationsInstancesListRequest', response_type_name='ListInstancesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Instance.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}', http_method='PATCH', method_id='parallelstore.projects.locations.instances.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='instance', request_type_name='ParallelstoreProjectsLocationsInstancesPatchRequest', response_type_name='Operation', supports_download=False)