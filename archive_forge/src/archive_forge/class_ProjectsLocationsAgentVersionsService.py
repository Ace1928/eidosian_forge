from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentVersionsService(base_api.BaseApiService):
    """Service class for the projects_locations_agent_versions resource."""
    _NAME = 'projects_locations_agent_versions'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentVersionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an agent version. The new version points to the agent instance in the "default" environment.

      Args:
        request: (DialogflowProjectsLocationsAgentVersionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Version) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/versions', http_method='POST', method_id='dialogflow.projects.locations.agent.versions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/versions', request_field='googleCloudDialogflowV2Version', request_type_name='DialogflowProjectsLocationsAgentVersionsCreateRequest', response_type_name='GoogleCloudDialogflowV2Version', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete the specified agent version.

      Args:
        request: (DialogflowProjectsLocationsAgentVersionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/versions/{versionsId}', http_method='DELETE', method_id='dialogflow.projects.locations.agent.versions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentVersionsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified agent version.

      Args:
        request: (DialogflowProjectsLocationsAgentVersionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Version) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/versions/{versionsId}', http_method='GET', method_id='dialogflow.projects.locations.agent.versions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentVersionsGetRequest', response_type_name='GoogleCloudDialogflowV2Version', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all versions of the specified agent.

      Args:
        request: (DialogflowProjectsLocationsAgentVersionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListVersionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/versions', http_method='GET', method_id='dialogflow.projects.locations.agent.versions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/versions', request_field='', request_type_name='DialogflowProjectsLocationsAgentVersionsListRequest', response_type_name='GoogleCloudDialogflowV2ListVersionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified agent version. Note that this method does not allow you to update the state of the agent the given version points to. It allows you to update only mutable properties of the version resource.

      Args:
        request: (DialogflowProjectsLocationsAgentVersionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Version) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/versions/{versionsId}', http_method='PATCH', method_id='dialogflow.projects.locations.agent.versions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Version', request_type_name='DialogflowProjectsLocationsAgentVersionsPatchRequest', response_type_name='GoogleCloudDialogflowV2Version', supports_download=False)