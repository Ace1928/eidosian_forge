from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsLocationsBatchesService(base_api.BaseApiService):
    """Service class for the projects_locations_batches resource."""
    _NAME = 'projects_locations_batches'

    def __init__(self, client):
        super(DataprocV1.ProjectsLocationsBatchesService, self).__init__(client)
        self._upload_configs = {}

    def Analyze(self, request, global_params=None):
        """Analyze a Batch for possible recommendations and insights.

      Args:
        request: (DataprocProjectsLocationsBatchesAnalyzeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Analyze')
        return self._RunMethod(config, request, global_params=global_params)
    Analyze.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batches/{batchesId}:analyze', http_method='POST', method_id='dataproc.projects.locations.batches.analyze', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:analyze', request_field='analyzeBatchRequest', request_type_name='DataprocProjectsLocationsBatchesAnalyzeRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a batch workload that executes asynchronously.

      Args:
        request: (DataprocProjectsLocationsBatchesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batches', http_method='POST', method_id='dataproc.projects.locations.batches.create', ordered_params=['parent'], path_params=['parent'], query_params=['batchId', 'requestId'], relative_path='v1/{+parent}/batches', request_field='batch', request_type_name='DataprocProjectsLocationsBatchesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the batch workload resource. If the batch is not in a CANCELLED, SUCCEEDED or FAILED State, the delete operation fails and the response returns FAILED_PRECONDITION.

      Args:
        request: (DataprocProjectsLocationsBatchesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batches/{batchesId}', http_method='DELETE', method_id='dataproc.projects.locations.batches.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsBatchesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the batch workload resource representation.

      Args:
        request: (DataprocProjectsLocationsBatchesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Batch) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batches/{batchesId}', http_method='GET', method_id='dataproc.projects.locations.batches.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsBatchesGetRequest', response_type_name='Batch', supports_download=False)

    def List(self, request, global_params=None):
        """Lists batch workloads.

      Args:
        request: (DataprocProjectsLocationsBatchesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBatchesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/batches', http_method='GET', method_id='dataproc.projects.locations.batches.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/batches', request_field='', request_type_name='DataprocProjectsLocationsBatchesListRequest', response_type_name='ListBatchesResponse', supports_download=False)