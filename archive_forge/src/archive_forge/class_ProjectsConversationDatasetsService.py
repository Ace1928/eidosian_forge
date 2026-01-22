from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsConversationDatasetsService(base_api.BaseApiService):
    """Service class for the projects_conversationDatasets resource."""
    _NAME = 'projects_conversationDatasets'

    def __init__(self, client):
        super(DialogflowV2.ProjectsConversationDatasetsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves the specified conversation dataset.

      Args:
        request: (DialogflowProjectsConversationDatasetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationDataset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationDatasets/{conversationDatasetsId}', http_method='GET', method_id='dialogflow.projects.conversationDatasets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsConversationDatasetsGetRequest', response_type_name='GoogleCloudDialogflowV2ConversationDataset', supports_download=False)

    def ImportConversationData(self, request, global_params=None):
        """Import data into the specified conversation dataset. Note that it is not allowed to import data to a conversation dataset that already has data in it. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: ImportConversationDataOperationMetadata - `response`: ImportConversationDataOperationResponse.

      Args:
        request: (DialogflowProjectsConversationDatasetsImportConversationDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ImportConversationData')
        return self._RunMethod(config, request, global_params=global_params)
    ImportConversationData.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationDatasets/{conversationDatasetsId}:importConversationData', http_method='POST', method_id='dialogflow.projects.conversationDatasets.importConversationData', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:importConversationData', request_field='googleCloudDialogflowV2ImportConversationDataRequest', request_type_name='DialogflowProjectsConversationDatasetsImportConversationDataRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all conversation datasets in the specified project and location.

      Args:
        request: (DialogflowProjectsConversationDatasetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListConversationDatasetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationDatasets', http_method='GET', method_id='dialogflow.projects.conversationDatasets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/conversationDatasets', request_field='', request_type_name='DialogflowProjectsConversationDatasetsListRequest', response_type_name='GoogleCloudDialogflowV2ListConversationDatasetsResponse', supports_download=False)