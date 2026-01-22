from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDatasetsSavedQueriesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_savedQueries resource."""
    _NAME = 'projects_locations_datasets_savedQueries'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDatasetsSavedQueriesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a SavedQuery.

      Args:
        request: (AiplatformProjectsLocationsDatasetsSavedQueriesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/savedQueries/{savedQueriesId}', http_method='DELETE', method_id='aiplatform.projects.locations.datasets.savedQueries.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsSavedQueriesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SavedQueries in a Dataset.

      Args:
        request: (AiplatformProjectsLocationsDatasetsSavedQueriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListSavedQueriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/savedQueries', http_method='GET', method_id='aiplatform.projects.locations.datasets.savedQueries.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'readMask'], relative_path='v1/{+parent}/savedQueries', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsSavedQueriesListRequest', response_type_name='GoogleCloudAiplatformV1ListSavedQueriesResponse', supports_download=False)