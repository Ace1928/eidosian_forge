from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDatasetsDatasetVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_datasetVersions resource."""
    _NAME = 'projects_locations_datasets_datasetVersions'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDatasetsDatasetVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a version from a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDatasetVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/datasetVersions', http_method='POST', method_id='aiplatform.projects.locations.datasets.datasetVersions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/datasetVersions', request_field='googleCloudAiplatformV1DatasetVersion', request_type_name='AiplatformProjectsLocationsDatasetsDatasetVersionsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Dataset version.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDatasetVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/datasetVersions/{datasetVersionsId}', http_method='DELETE', method_id='aiplatform.projects.locations.datasets.datasetVersions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDatasetVersionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Dataset version.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDatasetVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1DatasetVersion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/datasetVersions/{datasetVersionsId}', http_method='GET', method_id='aiplatform.projects.locations.datasets.datasetVersions.get', ordered_params=['name'], path_params=['name'], query_params=['readMask'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDatasetVersionsGetRequest', response_type_name='GoogleCloudAiplatformV1DatasetVersion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DatasetVersions in a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDatasetVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListDatasetVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/datasetVersions', http_method='GET', method_id='aiplatform.projects.locations.datasets.datasetVersions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/datasetVersions', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDatasetVersionsListRequest', response_type_name='GoogleCloudAiplatformV1ListDatasetVersionsResponse', supports_download=False)

    def Restore(self, request, global_params=None):
        """Restores a dataset version.

      Args:
        request: (AiplatformProjectsLocationsDatasetsDatasetVersionsRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/datasetVersions/{datasetVersionsId}:restore', http_method='GET', method_id='aiplatform.projects.locations.datasets.datasetVersions.restore', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:restore', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsDatasetVersionsRestoreRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)