from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataproc.v1 import dataproc_v1_messages as messages
class ProjectsLocationsSessionsService(base_api.BaseApiService):
    """Service class for the projects_locations_sessions resource."""
    _NAME = 'projects_locations_sessions'

    def __init__(self, client):
        super(DataprocV1.ProjectsLocationsSessionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an interactive session asynchronously.

      Args:
        request: (DataprocProjectsLocationsSessionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions', http_method='POST', method_id='dataproc.projects.locations.sessions.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'sessionId'], relative_path='v1/{+parent}/sessions', request_field='session', request_type_name='DataprocProjectsLocationsSessionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the interactive session resource. If the session is not in terminal state, it is terminated, and then deleted.

      Args:
        request: (DataprocProjectsLocationsSessionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}', http_method='DELETE', method_id='dataproc.projects.locations.sessions.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsSessionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the resource representation for an interactive session.

      Args:
        request: (DataprocProjectsLocationsSessionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Session) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}', http_method='GET', method_id='dataproc.projects.locations.sessions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DataprocProjectsLocationsSessionsGetRequest', response_type_name='Session', supports_download=False)

    def InjectCredentials(self, request, global_params=None):
        """Inject Credentials in the interactive session.

      Args:
        request: (DataprocProjectsLocationsSessionsInjectCredentialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('InjectCredentials')
        return self._RunMethod(config, request, global_params=global_params)
    InjectCredentials.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}:injectCredentials', http_method='POST', method_id='dataproc.projects.locations.sessions.injectCredentials', ordered_params=['session'], path_params=['session'], query_params=[], relative_path='v1/{+session}:injectCredentials', request_field='injectSessionCredentialsRequest', request_type_name='DataprocProjectsLocationsSessionsInjectCredentialsRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists interactive sessions.

      Args:
        request: (DataprocProjectsLocationsSessionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSessionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions', http_method='GET', method_id='dataproc.projects.locations.sessions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/sessions', request_field='', request_type_name='DataprocProjectsLocationsSessionsListRequest', response_type_name='ListSessionsResponse', supports_download=False)

    def Terminate(self, request, global_params=None):
        """Terminates the interactive session.

      Args:
        request: (DataprocProjectsLocationsSessionsTerminateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Terminate')
        return self._RunMethod(config, request, global_params=global_params)
    Terminate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sessions/{sessionsId}:terminate', http_method='POST', method_id='dataproc.projects.locations.sessions.terminate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:terminate', request_field='terminateSessionRequest', request_type_name='DataprocProjectsLocationsSessionsTerminateRequest', response_type_name='Operation', supports_download=False)