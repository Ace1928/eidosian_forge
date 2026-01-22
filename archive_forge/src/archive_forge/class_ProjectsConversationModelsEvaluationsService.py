from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsConversationModelsEvaluationsService(base_api.BaseApiService):
    """Service class for the projects_conversationModels_evaluations resource."""
    _NAME = 'projects_conversationModels_evaluations'

    def __init__(self, client):
        super(DialogflowV2.ProjectsConversationModelsEvaluationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets an evaluation of conversation model.

      Args:
        request: (DialogflowProjectsConversationModelsEvaluationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationModelEvaluation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels/{conversationModelsId}/evaluations/{evaluationsId}', http_method='GET', method_id='dialogflow.projects.conversationModels.evaluations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsConversationModelsEvaluationsGetRequest', response_type_name='GoogleCloudDialogflowV2ConversationModelEvaluation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists evaluations of a conversation model.

      Args:
        request: (DialogflowProjectsConversationModelsEvaluationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListConversationModelEvaluationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels/{conversationModelsId}/evaluations', http_method='GET', method_id='dialogflow.projects.conversationModels.evaluations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/evaluations', request_field='', request_type_name='DialogflowProjectsConversationModelsEvaluationsListRequest', response_type_name='GoogleCloudDialogflowV2ListConversationModelEvaluationsResponse', supports_download=False)