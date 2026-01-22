from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsMigratableResourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_migratableResources resource."""
    _NAME = 'projects_locations_migratableResources'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsMigratableResourcesService, self).__init__(client)
        self._upload_configs = {}

    def BatchMigrate(self, request, global_params=None):
        """Batch migrates resources from ml.googleapis.com, automl.googleapis.com, and datalabeling.googleapis.com to Vertex AI.

      Args:
        request: (AiplatformProjectsLocationsMigratableResourcesBatchMigrateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchMigrate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchMigrate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/migratableResources:batchMigrate', http_method='POST', method_id='aiplatform.projects.locations.migratableResources.batchMigrate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/migratableResources:batchMigrate', request_field='googleCloudAiplatformV1BatchMigrateResourcesRequest', request_type_name='AiplatformProjectsLocationsMigratableResourcesBatchMigrateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Search(self, request, global_params=None):
        """Searches all of the resources in automl.googleapis.com, datalabeling.googleapis.com and ml.googleapis.com that can be migrated to Vertex AI's given location.

      Args:
        request: (AiplatformProjectsLocationsMigratableResourcesSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1SearchMigratableResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/migratableResources:search', http_method='POST', method_id='aiplatform.projects.locations.migratableResources.search', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/migratableResources:search', request_field='googleCloudAiplatformV1SearchMigratableResourcesRequest', request_type_name='AiplatformProjectsLocationsMigratableResourcesSearchRequest', response_type_name='GoogleCloudAiplatformV1SearchMigratableResourcesResponse', supports_download=False)