from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsTensorboardsExperimentsRunsTimeSeriesService(base_api.BaseApiService):
    """Service class for the projects_locations_tensorboards_experiments_runs_timeSeries resource."""
    _NAME = 'projects_locations_tensorboards_experiments_runs_timeSeries'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsTensorboardsExperimentsRunsTimeSeriesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a TensorboardTimeSeries.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TensorboardTimeSeries) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries', http_method='POST', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.create', ordered_params=['parent'], path_params=['parent'], query_params=['tensorboardTimeSeriesId'], relative_path='v1/{+parent}/timeSeries', request_field='googleCloudAiplatformV1TensorboardTimeSeries', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesCreateRequest', response_type_name='GoogleCloudAiplatformV1TensorboardTimeSeries', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a TensorboardTimeSeries.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries/{timeSeriesId}', http_method='DELETE', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ExportTensorboardTimeSeries(self, request, global_params=None):
        """Exports a TensorboardTimeSeries' data. Data is returned in paginated responses.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesExportTensorboardTimeSeriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ExportTensorboardTimeSeriesDataResponse) The response message.
      """
        config = self.GetMethodConfig('ExportTensorboardTimeSeries')
        return self._RunMethod(config, request, global_params=global_params)
    ExportTensorboardTimeSeries.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries/{timeSeriesId}:exportTensorboardTimeSeries', http_method='POST', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.exportTensorboardTimeSeries', ordered_params=['tensorboardTimeSeries'], path_params=['tensorboardTimeSeries'], query_params=[], relative_path='v1/{+tensorboardTimeSeries}:exportTensorboardTimeSeries', request_field='googleCloudAiplatformV1ExportTensorboardTimeSeriesDataRequest', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesExportTensorboardTimeSeriesRequest', response_type_name='GoogleCloudAiplatformV1ExportTensorboardTimeSeriesDataResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a TensorboardTimeSeries.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TensorboardTimeSeries) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries/{timeSeriesId}', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesGetRequest', response_type_name='GoogleCloudAiplatformV1TensorboardTimeSeries', supports_download=False)

    def List(self, request, global_params=None):
        """Lists TensorboardTimeSeries in a Location.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListTensorboardTimeSeriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/timeSeries', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesListRequest', response_type_name='GoogleCloudAiplatformV1ListTensorboardTimeSeriesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a TensorboardTimeSeries.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1TensorboardTimeSeries) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries/{timeSeriesId}', http_method='PATCH', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1TensorboardTimeSeries', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesPatchRequest', response_type_name='GoogleCloudAiplatformV1TensorboardTimeSeries', supports_download=False)

    def Read(self, request, global_params=None):
        """Reads a TensorboardTimeSeries' data. By default, if the number of data points stored is less than 1000, all data is returned. Otherwise, 1000 data points is randomly selected from this time series and returned. This value can be changed by changing max_data_points, which can't be greater than 10k.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardTimeSeriesDataResponse) The response message.
      """
        config = self.GetMethodConfig('Read')
        return self._RunMethod(config, request, global_params=global_params)
    Read.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries/{timeSeriesId}:read', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.read', ordered_params=['tensorboardTimeSeries'], path_params=['tensorboardTimeSeries'], query_params=['filter', 'maxDataPoints'], relative_path='v1/{+tensorboardTimeSeries}:read', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadRequest', response_type_name='GoogleCloudAiplatformV1ReadTensorboardTimeSeriesDataResponse', supports_download=False)

    def ReadBlobData(self, request, global_params=None):
        """Gets bytes of TensorboardBlobs. This is to allow reading blob data stored in consumer project's Cloud Storage bucket without users having to obtain Cloud Storage access permission.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadBlobDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardBlobDataResponse) The response message.
      """
        config = self.GetMethodConfig('ReadBlobData')
        return self._RunMethod(config, request, global_params=global_params)
    ReadBlobData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/tensorboards/{tensorboardsId}/experiments/{experimentsId}/runs/{runsId}/timeSeries/{timeSeriesId}:readBlobData', http_method='GET', method_id='aiplatform.projects.locations.tensorboards.experiments.runs.timeSeries.readBlobData', ordered_params=['timeSeries'], path_params=['timeSeries'], query_params=['blobIds'], relative_path='v1/{+timeSeries}:readBlobData', request_field='', request_type_name='AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadBlobDataRequest', response_type_name='GoogleCloudAiplatformV1ReadTensorboardBlobDataResponse', supports_download=False)