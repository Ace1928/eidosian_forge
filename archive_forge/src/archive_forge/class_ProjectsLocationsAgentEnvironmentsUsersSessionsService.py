from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentEnvironmentsUsersSessionsService(base_api.BaseApiService):
    """Service class for the projects_locations_agent_environments_users_sessions resource."""
    _NAME = 'projects_locations_agent_environments_users_sessions'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentEnvironmentsUsersSessionsService, self).__init__(client)
        self._upload_configs = {}

    def DeleteContexts(self, request, global_params=None):
        """Deletes all active contexts in the specified session.

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsDeleteContextsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('DeleteContexts')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteContexts.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}/contexts', http_method='DELETE', method_id='dialogflow.projects.locations.agent.environments.users.sessions.deleteContexts', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/contexts', request_field='', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsDeleteContextsRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def DetectIntent(self, request, global_params=None):
        """Processes a natural language query and returns structured, actionable data as a result. This method is not idempotent, because it may cause contexts and session entity types to be updated, which in turn might affect results of future queries. If you might use [Agent Assist](https://cloud.google.com/dialogflow/docs/#aa) or other CCAI products now or in the future, consider using AnalyzeContent instead of `DetectIntent`. `AnalyzeContent` has additional functionality for Agent Assist and other CCAI products. Note: Always use agent versions for production traffic. See [Versions and environments](https://cloud.google.com/dialogflow/es/docs/agents-versions).

      Args:
        request: (DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsDetectIntentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2DetectIntentResponse) The response message.
      """
        config = self.GetMethodConfig('DetectIntent')
        return self._RunMethod(config, request, global_params=global_params)
    DetectIntent.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/environments/{environmentsId}/users/{usersId}/sessions/{sessionsId}:detectIntent', http_method='POST', method_id='dialogflow.projects.locations.agent.environments.users.sessions.detectIntent', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v2/{+session}:detectIntent', request_field='googleCloudDialogflowV2DetectIntentRequest', request_type_name='DialogflowProjectsLocationsAgentEnvironmentsUsersSessionsDetectIntentRequest', response_type_name='GoogleCloudDialogflowV2DetectIntentResponse', supports_download=False)