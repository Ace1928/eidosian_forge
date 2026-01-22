from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsMetadataStoresContextsService(base_api.BaseApiService):
    """Service class for the projects_locations_metadataStores_contexts resource."""
    _NAME = 'projects_locations_metadataStores_contexts'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsMetadataStoresContextsService, self).__init__(client)
        self._upload_configs = {}

    def AddContextArtifactsAndExecutions(self, request, global_params=None):
        """Adds a set of Artifacts and Executions to a Context. If any of the Artifacts or Executions have already been added to a Context, they are simply skipped.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsAddContextArtifactsAndExecutionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AddContextArtifactsAndExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('AddContextArtifactsAndExecutions')
        return self._RunMethod(config, request, global_params=global_params)
    AddContextArtifactsAndExecutions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}:addContextArtifactsAndExecutions', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.addContextArtifactsAndExecutions', ordered_params=['context'], path_params=['context'], query_params=[], relative_path='v1/{+context}:addContextArtifactsAndExecutions', request_field='googleCloudAiplatformV1AddContextArtifactsAndExecutionsRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsAddContextArtifactsAndExecutionsRequest', response_type_name='GoogleCloudAiplatformV1AddContextArtifactsAndExecutionsResponse', supports_download=False)

    def AddContextChildren(self, request, global_params=None):
        """Adds a set of Contexts as children to a parent Context. If any of the child Contexts have already been added to the parent Context, they are simply skipped. If this call would create a cycle or cause any Context to have more than 10 parents, the request will fail with an INVALID_ARGUMENT error.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsAddContextChildrenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AddContextChildrenResponse) The response message.
      """
        config = self.GetMethodConfig('AddContextChildren')
        return self._RunMethod(config, request, global_params=global_params)
    AddContextChildren.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}:addContextChildren', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.addContextChildren', ordered_params=['context'], path_params=['context'], query_params=[], relative_path='v1/{+context}:addContextChildren', request_field='googleCloudAiplatformV1AddContextChildrenRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsAddContextChildrenRequest', response_type_name='GoogleCloudAiplatformV1AddContextChildrenResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a Context associated with a MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Context) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.create', ordered_params=['parent'], path_params=['parent'], query_params=['contextId'], relative_path='v1/{+parent}/contexts', request_field='googleCloudAiplatformV1Context', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsCreateRequest', response_type_name='GoogleCloudAiplatformV1Context', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a stored Context.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}', http_method='DELETE', method_id='aiplatform.projects.locations.metadataStores.contexts.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific Context.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Context) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.contexts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsGetRequest', response_type_name='GoogleCloudAiplatformV1Context', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Contexts on the MetadataStore.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListContextsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.contexts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/contexts', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsListRequest', response_type_name='GoogleCloudAiplatformV1ListContextsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a stored Context.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Context) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}', http_method='PATCH', method_id='aiplatform.projects.locations.metadataStores.contexts.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask'], relative_path='v1/{+name}', request_field='googleCloudAiplatformV1Context', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsPatchRequest', response_type_name='GoogleCloudAiplatformV1Context', supports_download=False)

    def Purge(self, request, global_params=None):
        """Purges Contexts.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsPurgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Purge')
        return self._RunMethod(config, request, global_params=global_params)
    Purge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts:purge', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.purge', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/contexts:purge', request_field='googleCloudAiplatformV1PurgeContextsRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsPurgeRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def QueryContextLineageSubgraph(self, request, global_params=None):
        """Retrieves Artifacts and Executions within the specified Context, connected by Event edges and returned as a LineageSubgraph.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsQueryContextLineageSubgraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1LineageSubgraph) The response message.
      """
        config = self.GetMethodConfig('QueryContextLineageSubgraph')
        return self._RunMethod(config, request, global_params=global_params)
    QueryContextLineageSubgraph.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}:queryContextLineageSubgraph', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.contexts.queryContextLineageSubgraph', ordered_params=['context'], path_params=['context'], query_params=[], relative_path='v1/{+context}:queryContextLineageSubgraph', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsQueryContextLineageSubgraphRequest', response_type_name='GoogleCloudAiplatformV1LineageSubgraph', supports_download=False)

    def RemoveContextChildren(self, request, global_params=None):
        """Remove a set of children contexts from a parent Context. If any of the child Contexts were NOT added to the parent Context, they are simply skipped.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsRemoveContextChildrenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1RemoveContextChildrenResponse) The response message.
      """
        config = self.GetMethodConfig('RemoveContextChildren')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveContextChildren.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}:removeContextChildren', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.removeContextChildren', ordered_params=['context'], path_params=['context'], query_params=[], relative_path='v1/{+context}:removeContextChildren', request_field='googleCloudAiplatformV1RemoveContextChildrenRequest', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsRemoveContextChildrenRequest', response_type_name='GoogleCloudAiplatformV1RemoveContextChildrenResponse', supports_download=False)