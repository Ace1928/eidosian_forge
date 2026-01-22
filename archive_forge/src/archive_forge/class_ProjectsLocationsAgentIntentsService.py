from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentIntentsService(base_api.BaseApiService):
    """Service class for the projects_locations_agent_intents resource."""
    _NAME = 'projects_locations_agent_intents'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentIntentsService, self).__init__(client)
        self._upload_configs = {}

    def BatchDelete(self, request, global_params=None):
        """Deletes intents in the specified agent. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsBatchDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchDelete')
        return self._RunMethod(config, request, global_params=global_params)
    BatchDelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents:batchDelete', http_method='POST', method_id='dialogflow.projects.locations.agent.intents.batchDelete', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/intents:batchDelete', request_field='googleCloudDialogflowV2BatchDeleteIntentsRequest', request_type_name='DialogflowProjectsLocationsAgentIntentsBatchDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def BatchUpdate(self, request, global_params=None):
        """Updates/Creates multiple intents in the specified agent. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: BatchUpdateIntentsResponse Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsBatchUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('BatchUpdate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchUpdate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents:batchUpdate', http_method='POST', method_id='dialogflow.projects.locations.agent.intents.batchUpdate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/intents:batchUpdate', request_field='googleCloudDialogflowV2BatchUpdateIntentsRequest', request_type_name='DialogflowProjectsLocationsAgentIntentsBatchUpdateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates an intent in the specified agent. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Intent) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents', http_method='POST', method_id='dialogflow.projects.locations.agent.intents.create', ordered_params=['parent'], path_params=['parent'], query_params=['intentView', 'languageCode'], relative_path='v2/{+parent}/intents', request_field='googleCloudDialogflowV2Intent', request_type_name='DialogflowProjectsLocationsAgentIntentsCreateRequest', response_type_name='GoogleCloudDialogflowV2Intent', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified intent and its direct or indirect followup intents. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents/{intentsId}', http_method='DELETE', method_id='dialogflow.projects.locations.agent.intents.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentIntentsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified intent.

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Intent) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents/{intentsId}', http_method='GET', method_id='dialogflow.projects.locations.agent.intents.get', ordered_params=['name'], path_params=['name'], query_params=['intentView', 'languageCode'], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentIntentsGetRequest', response_type_name='GoogleCloudDialogflowV2Intent', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all intents in the specified agent.

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListIntentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents', http_method='GET', method_id='dialogflow.projects.locations.agent.intents.list', ordered_params=['parent'], path_params=['parent'], query_params=['intentView', 'languageCode', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/intents', request_field='', request_type_name='DialogflowProjectsLocationsAgentIntentsListRequest', response_type_name='GoogleCloudDialogflowV2ListIntentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified intent. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentIntentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Intent) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/intents/{intentsId}', http_method='PATCH', method_id='dialogflow.projects.locations.agent.intents.patch', ordered_params=['name'], path_params=['name'], query_params=['intentView', 'languageCode', 'updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Intent', request_type_name='DialogflowProjectsLocationsAgentIntentsPatchRequest', response_type_name='GoogleCloudDialogflowV2Intent', supports_download=False)