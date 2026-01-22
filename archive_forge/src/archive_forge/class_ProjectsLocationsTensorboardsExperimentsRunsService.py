from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsTensorboardsExperimentsRunsService(base_api.BaseApiService):
    """Service class for the projects_locations_tensorboards_experiments_runs resource."""
    _NAME = 'projects_locations_tensorboards_experiments_runs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsTensorboardsExperimentsRunsService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Batch create TensorboardRuns.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchCreateTensorboardRunsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs:batchCreate', http_method='POST', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/runs:batchCreate', request_field='googleCloudAiplatformV1BatchCreateTensorboardRunsRequest', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsBatchCreateRequest', response_type_name='GoogleCloudAiplatformV1BatchCreateTensorboardRunsResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a TensorboardRun.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TensorboardRun) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs', http_method='POST', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.create', ordered_params=['parent'], path_params=['parent'], query_params=['tensorboardRunId'], relative_path='v1/{+parent}/runs', request_field='googleCloudAiplatformV1TensorboardRun', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsCreateRequest', response_type_name='GoogleCloudAiplatformV1TensorboardRun', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TensorboardRun.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}', http_method='DELETE', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a TensorboardRun.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TensorboardRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsGetRequest', response_type_name='GoogleCloudAiplatformV1TensorboardRun', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TensorboardRuns in a Location.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListTensorboardRunsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/runs', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsListRequest', response_type_name='GoogleCloudAiplatformV1ListTensorboardRunsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a TensorboardRun.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TensorboardRun) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}', http_method='PATCH', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1TensorboardRun', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsPatchRequest', response_type_name='GoogleCloudAiplatformV1TensorboardRun', supports_download=False)

    def Write(self, request, global_params=None):
        """Write time series data points into multiple TensorboardTimeSeries under a TensorboardRun. If any data fail to be ingested, an error is returned.

      Args:
        request: (GoogleCloudAiplatformV1WriteTensorboardRunDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1WriteTensorboardRunDataResponse) The response message.
      """
        config = self.GetMethodConfig('Write')
        return self._RunMethod(config, request, global_params=global_params)
    Write.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}:write', http_method='POST', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.write', ordered_params=['tensorboardRun'], path_params=['tensorboardRun'], query_params=[], relative_path='v1/{+tensorboardRun}:write', request_field='<request>', request_type_name='GoogleCloudAiplatformV1WriteTensorboardRunDataRequest', response_type_name='GoogleCloudAiplatformV1WriteTensorboardRunDataResponse', supports_download=False)