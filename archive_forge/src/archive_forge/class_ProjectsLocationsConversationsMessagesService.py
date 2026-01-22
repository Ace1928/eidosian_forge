from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsConversationsMessagesService(base_api.BaseApiService):
    """Service class for the projects_locations_conversations_messages resource."""
    _NAME = 'projects_locations_conversations_messages'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsConversationsMessagesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists messages that belong to a given conversation. `messages` are ordered by `create_time` in descending order. To fetch updates without duplication, send request with filter `create_time_epoch_microseconds > [first item's create_time of previous request]` and empty page_token.

      Args:
        request: (DialogflowProjectsLocationsConversationsMessagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListMessagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/conversations/{conversationsId}/messages', http_method='GET', method_id='dialogflow.projects.locations.conversations.messages.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/messages', request_field='', request_type_name='DialogflowProjectsLocationsConversationsMessagesListRequest', response_type_name='GoogleCloudDialogflowV2ListMessagesResponse', supports_download=False)