from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsMetadataStoresArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_metadataStores_artifacts resource."""
    _NAME = 'projects_locations_metadataStores_artifacts'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsMetadataStoresArtifactsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an Artifact associated with a MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Artifact) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.artifacts.create', ordered_params=['parent'], path_params=['parent'], query_params=['artifactId'], relative_path='v1/{+parent}/artifacts', request_field='googleCloudAiplatformV1Artifact', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsCreateRequest', response_type_name='GoogleCloudAiplatformV1Artifact', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Artifact.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts/{artifactsId}', http_method='DELETE', method_id='aiplatform.projects.locations.metadataStores.artifacts.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific Artifact.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Artifact) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts/{artifactsId}', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.artifacts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsGetRequest', response_type_name='GoogleCloudAiplatformV1Artifact', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Artifacts in the MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListArtifactsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.artifacts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/artifacts', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsListRequest', response_type_name='GoogleCloudAiplatformV1ListArtifactsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a stored Artifact.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Artifact) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts/{artifactsId}', http_method='PATCH', method_id='aiplatform.projects.locations.metadataStores.artifacts.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Artifact', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsPatchRequest', response_type_name='GoogleCloudAiplatformV1Artifact', supports_download=False)

    def Purge(self, request, global_params=None):
        """Purges Artifacts.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsPurgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Purge')
        return self._RunMethod(config, request, global_params=global_params)
    Purge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts:purge', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.artifacts.purge', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/artifacts:purge', request_field='googleCloudAiplatformV1PurgeArtifactsRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsPurgeRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def QueryArtifactLineageSubgraph(self, request, global_params=None):
        """Retrieves lineage of an Artifact represented through Artifacts and Executions connected by Event edges and returned as a LineageSubgraph.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresArtifactsQueryArtifactLineageSubgraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1LineageSubgraph) The response message.
      """
        config = self.GetMethodConfig('QueryArtifactLineageSubgraph')
        return self._RunMethod(config, request, global_params=global_params)
    QueryArtifactLineageSubgraph.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/artifacts/{artifactsId}:queryArtifactLineageSubgraph', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.artifacts.queryArtifactLineageSubgraph', ordered_params=['artifact'], path_params=['artifact'], query_params=['filter', 'maxHops'], relative_path='v1/{+artifact}:queryArtifactLineageSubgraph', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresArtifactsQueryArtifactLineageSubgraphRequest', response_type_name='GoogleCloudAiplatformV1LineageSubgraph', supports_download=False)