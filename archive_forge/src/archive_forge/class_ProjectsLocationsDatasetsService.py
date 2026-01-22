from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDatasetsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets resource."""
    _NAME = 'projects_locations_datasets'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDatasetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets', http_method='POST', method_id='aiplatform.projects.locations.datasets.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/datasets', request_field='googleCloudAiplatformV1Dataset', request_type_name='AiplatformProjectsLocationsDatasetsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}', http_method='DELETE', method_id='aiplatform.projects.locations.datasets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports data from a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}:export', http_method='POST', method_id='aiplatform.projects.locations.datasets.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:export', request_field='googleCloudAiplatformV1ExportDataRequest', request_type_name='AiplatformProjectsLocationsDatasetsExportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Dataset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}', http_method='GET', method_id='aiplatform.projects.locations.datasets.get', ordered_params=['name'], path_params=['name'], query_params=['readMask'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsGetRequest', response_type_name='GoogleCloudAiplatformV1Dataset', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports data into a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}:import', http_method='POST', method_id='aiplatform.projects.locations.datasets.import', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:import', request_field='googleCloudAiplatformV1ImportDataRequest', request_type_name='AiplatformProjectsLocationsDatasetsImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Datasets in a Location.

      Args:
        request: (AiplatformProjectsLocationsDatasetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListDatasetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets', http_method='GET', method_id='aiplatform.projects.locations.datasets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/datasets', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsListRequest', response_type_name='GoogleCloudAiplatformV1ListDatasetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Dataset) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}', http_method='PATCH', method_id='aiplatform.projects.locations.datasets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Dataset', request_type_name='AiplatformProjectsLocationsDatasetsPatchRequest', response_type_name='GoogleCloudAiplatformV1Dataset', supports_download=False)

    def SearchDataItems(self, request, global_params=None):
        """Searches DataItems in a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsSearchDataItemsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchDataItemsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchDataItems')
        return self._RunMethod(config, request, global_params=global_params)
    SearchDataItems.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}:searchDataItems', http_method='GET', method_id='aiplatform.projects.locations.datasets.searchDataItems', ordered_params=['dataset'], path_params=['dataset'], query_params=['annotationFilters', 'annotationsFilter', 'annotationsLimit', 'dataItemFilter', 'dataLabelingJob', 'fieldMask', 'orderBy', 'orderByAnnotation_orderBy', 'orderByAnnotation_savedQuery', 'orderByDataItem', 'pageSize', 'pageToken', 'savedQuery'], relative_path='v1/{+dataset}:searchDataItems', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsSearchDataItemsRequest', response_type_name='GoogleCloudAiplatformV1SearchDataItemsResponse', supports_download=False)