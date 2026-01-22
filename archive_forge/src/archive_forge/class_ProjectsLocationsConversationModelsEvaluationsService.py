from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsConversationModelsEvaluationsService(base_api.BaseApiService):
    """Service class for the projects_locations_conversationModels_evaluations resource."""
    _NAME = 'projects_locations_conversationModels_evaluations'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsConversationModelsEvaluationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates evaluation of a conversation model.

      Args:
        request: (DialogflowProjectsLocationsConversationModelsEvaluationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationModels/{conversationModelsId}/evaluations', http_method='POST', method_id='dialogflow.projects.locations.conversationModels.evaluations.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/evaluations', request_field='googleCloudDialogflowV2CreateConversationModelEvaluationRequest', request_type_name='DialogflowProjectsLocationsConversationModelsEvaluationsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an evaluation of conversation model.

      Args:
        request: (DialogflowProjectsLocationsConversationModelsEvaluationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationModelEvaluation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationModels/{conversationModelsId}/evaluations/{evaluationsId}', http_method='GET', method_id='dialogflow.projects.locations.conversationModels.evaluations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsConversationModelsEvaluationsGetRequest', response_type_name='GoogleCloudDialogflowV2ConversationModelEvaluation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists evaluations of a conversation model.

      Args:
        request: (DialogflowProjectsLocationsConversationModelsEvaluationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListConversationModelEvaluationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversationModels/{conversationModelsId}/evaluations', http_method='GET', method_id='dialogflow.projects.locations.conversationModels.evaluations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/evaluations', request_field='', request_type_name='DialogflowProjectsLocationsConversationModelsEvaluationsListRequest', response_type_name='GoogleCloudDialogflowV2ListConversationModelEvaluationsResponse', supports_download=False)