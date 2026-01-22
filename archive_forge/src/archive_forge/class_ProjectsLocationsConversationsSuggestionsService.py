from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsConversationsSuggestionsService(base_api.BaseApiService):
    """Service class for the projects_locations_conversations_suggestions resource."""
    _NAME = 'projects_locations_conversations_suggestions'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsConversationsSuggestionsService, self).__init__(client)
        self._upload_configs = {}

    def SearchKnowledge(self, request, global_params=None):
        """Get answers for the given query based on knowledge documents.

      Args:
        request: (GoogleCloudDialogflowV2SearchKnowledgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SearchKnowledgeResponse) The response message.
      """
        config = self.GetMethodConfig('SearchKnowledge')
        return self._RunMethod(config, request, global_params=global_params)
    SearchKnowledge.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/suggestions:searchKnowledge', http_method='POST', method_id='dialogflow.projects.locations.conversations.suggestions.searchKnowledge', ordered_params=['conversation'], path_params=['conversation'], query_params=[], relative_path='v2/{+conversation}/suggestions:searchKnowledge', request_field='<request>', request_type_name='GoogleCloudDialogflowV2SearchKnowledgeRequest', response_type_name='GoogleCloudDialogflowV2SearchKnowledgeResponse', supports_download=False)

    def SuggestConversationSummary(self, request, global_params=None):
        """Suggests summary for a conversation based on specific historical messages. The range of the messages to be used for summary can be specified in the request.

      Args:
        request: (DialogflowProjectsLocationsConversationsSuggestionsSuggestConversationSummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SuggestConversationSummaryResponse) The response message.
      """
        config = self.GetMethodConfig('SuggestConversationSummary')
        return self._RunMethod(config, request, global_params=global_params)
    SuggestConversationSummary.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/suggestions:suggestConversationSummary', http_method='POST', method_id='dialogflow.projects.locations.conversations.suggestions.suggestConversationSummary', ordered_params=['conversation'], path_params=['conversation'], query_params=[], relative_path='v2/{+conversation}/suggestions:suggestConversationSummary', request_field='googleCloudDialogflowV2SuggestConversationSummaryRequest', request_type_name='DialogflowProjectsLocationsConversationsSuggestionsSuggestConversationSummaryRequest', response_type_name='GoogleCloudDialogflowV2SuggestConversationSummaryResponse', supports_download=False)