from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsRagCorporaService(base_api.BaseApiService):
    """Service class for the projects_locations_ragCorpora resource."""
    _NAME = 'projects_locations_ragCorpora'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsRagCorporaService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a RagCorpus.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora', http_method='POST', method_id='aiplatform.projects.locations.ragCorpora.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/ragCorpora', request_field='googleCloudAiplatformV1beta1RagCorpus', request_type_name='AiplatformProjectsLocationsRagCorporaCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a RagCorpus.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}', http_method='DELETE', method_id='aiplatform.projects.locations.ragCorpora.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsRagCorporaDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a RagCorpus.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1RagCorpus) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}', http_method='GET', method_id='aiplatform.projects.locations.ragCorpora.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsRagCorporaGetRequest', response_type_name='GoogleCloudAiplatformV1beta1RagCorpus', supports_download=False)

    def List(self, request, global_params=None):
        """Lists RagCorpora in a Location.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ListRagCorporaResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora', http_method='GET', method_id='aiplatform.projects.locations.ragCorpora.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/ragCorpora', request_field='', request_type_name='AiplatformProjectsLocationsRagCorporaListRequest', response_type_name='GoogleCloudAiplatformV1beta1ListRagCorporaResponse', supports_download=False)