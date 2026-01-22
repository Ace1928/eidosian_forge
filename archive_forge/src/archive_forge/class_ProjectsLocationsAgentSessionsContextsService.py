from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentSessionsContextsService(base_api.BaseApiService):
    """Service class for the projects_locations_agent_sessions_contexts resource."""
    _NAME = 'projects_locations_agent_sessions_contexts'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentSessionsContextsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a context. If the specified context already exists, overrides the context.

      Args:
        request: (DialogflowProjectsLocationsAgentSessionsContextsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Context) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/sessions/{sessionsId}/contexts', http_method='POST', method_id='dialogflow.projects.locations.agent.sessions.contexts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/contexts', request_field='googleCloudDialogflowV2Context', request_type_name='DialogflowProjectsLocationsAgentSessionsContextsCreateRequest', response_type_name='GoogleCloudDialogflowV2Context', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified context.

      Args:
        request: (DialogflowProjectsLocationsAgentSessionsContextsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/sessions/{sessionsId}/contexts/{contextsId}', http_method='DELETE', method_id='dialogflow.projects.locations.agent.sessions.contexts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentSessionsContextsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified context.

      Args:
        request: (DialogflowProjectsLocationsAgentSessionsContextsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Context) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/sessions/{sessionsId}/contexts/{contextsId}', http_method='GET', method_id='dialogflow.projects.locations.agent.sessions.contexts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentSessionsContextsGetRequest', response_type_name='GoogleCloudDialogflowV2Context', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all contexts in the specified session.

      Args:
        request: (DialogflowProjectsLocationsAgentSessionsContextsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListContextsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/sessions/{sessionsId}/contexts', http_method='GET', method_id='dialogflow.projects.locations.agent.sessions.contexts.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/contexts', request_field='', request_type_name='DialogflowProjectsLocationsAgentSessionsContextsListRequest', response_type_name='GoogleCloudDialogflowV2ListContextsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified context.

      Args:
        request: (DialogflowProjectsLocationsAgentSessionsContextsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Context) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/sessions/{sessionsId}/contexts/{contextsId}', http_method='PATCH', method_id='dialogflow.projects.locations.agent.sessions.contexts.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Context', request_type_name='DialogflowProjectsLocationsAgentSessionsContextsPatchRequest', response_type_name='GoogleCloudDialogflowV2Context', supports_download=False)