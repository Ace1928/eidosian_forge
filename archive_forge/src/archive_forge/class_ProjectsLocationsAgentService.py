from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsLocationsAgentService(base_api.BaseApiService):
    """Service class for the projects_locations_agent resource."""
    _NAME = 'projects_locations_agent'

    def __init__(self, client):
        super(DialogflowV2.ProjectsLocationsAgentService, self).__init__(client)
        self._upload_configs = {}

    def Export(self, request, global_params=None):
        """Exports the specified agent to a ZIP file. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: ExportAgentResponse.

      Args:
        request: (DialogflowProjectsLocationsAgentExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent:export', http_method='POST', method_id='dialogflow.projects.locations.agent.export', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:export', request_field='googleCloudDialogflowV2ExportAgentRequest', request_type_name='DialogflowProjectsLocationsAgentExportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def GetFulfillment(self, request, global_params=None):
        """Retrieves the fulfillment.

      Args:
        request: (DialogflowProjectsLocationsAgentGetFulfillmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Fulfillment) The response message.
      """
        config = self.GetMethodConfig('GetFulfillment')
        return self._RunMethod(config, request, global_params=global_params)
    GetFulfillment.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/fulfillment', http_method='GET', method_id='dialogflow.projects.locations.agent.getFulfillment', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsLocationsAgentGetFulfillmentRequest', response_type_name='GoogleCloudDialogflowV2Fulfillment', supports_download=False)

    def GetValidationResult(self, request, global_params=None):
        """Gets agent validation result. Agent validation is performed during training time and is updated automatically when training is completed.

      Args:
        request: (DialogflowProjectsLocationsAgentGetValidationResultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ValidationResult) The response message.
      """
        config = self.GetMethodConfig('GetValidationResult')
        return self._RunMethod(config, request, global_params=global_params)
    GetValidationResult.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/validationResult', http_method='GET', method_id='dialogflow.projects.locations.agent.getValidationResult', ordered_params=['parent'], path_params=['parent'], query_params=['languageCode'], relative_path='v2/{+parent}/agent/validationResult', request_field='', request_type_name='DialogflowProjectsLocationsAgentGetValidationResultRequest', response_type_name='GoogleCloudDialogflowV2ValidationResult', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports the specified agent from a ZIP file. Uploads new intents and entity types without deleting the existing ones. Intents and entity types with the same name are replaced with the new versions from ImportAgentRequest. After the import, the imported draft agent will be trained automatically (unless disabled in agent settings). However, once the import is done, training may not be completed yet. Please call TrainAgent and wait for the operation it returns in order to train explicitly. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) The operation only tracks when importing is complete, not when it is done training. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent:import', http_method='POST', method_id='dialogflow.projects.locations.agent.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:import', request_field='googleCloudDialogflowV2ImportAgentRequest', request_type_name='DialogflowProjectsLocationsAgentImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Restore(self, request, global_params=None):
        """Restores the specified agent from a ZIP file. Replaces the current agent version with a new one. All the intents and entity types in the older version are deleted. After the restore, the restored draft agent will be trained automatically (unless disabled in agent settings). However, once the restore is done, training may not be completed yet. Please call TrainAgent and wait for the operation it returns in order to train explicitly. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) The operation only tracks when restoring is complete, not when it is done training. Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent:restore', http_method='POST', method_id='dialogflow.projects.locations.agent.restore', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:restore', request_field='googleCloudDialogflowV2RestoreAgentRequest', request_type_name='DialogflowProjectsLocationsAgentRestoreRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Search(self, request, global_params=None):
        """Returns the list of agents. Since there is at most one conversational agent per project, this method is useful primarily for listing all agents across projects the caller has access to. One can achieve that with a wildcard project collection id "-". Refer to [List Sub-Collections](https://cloud.google.com/apis/design/design_patterns#list_sub-collections).

      Args:
        request: (DialogflowProjectsLocationsAgentSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2SearchAgentsResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent:search', http_method='GET', method_id='dialogflow.projects.locations.agent.search', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/agent:search', request_field='', request_type_name='DialogflowProjectsLocationsAgentSearchRequest', response_type_name='GoogleCloudDialogflowV2SearchAgentsResponse', supports_download=False)

    def Train(self, request, global_params=None):
        """Trains the specified agent. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: An empty [Struct message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#struct) - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty) Note: You should always train an agent prior to sending it queries. See the [training documentation](https://cloud.google.com/dialogflow/es/docs/training).

      Args:
        request: (DialogflowProjectsLocationsAgentTrainRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Train')
        return self._RunMethod(config, request, global_params=global_params)
    Train.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent:train', http_method='POST', method_id='dialogflow.projects.locations.agent.train', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/agent:train', request_field='googleCloudDialogflowV2TrainAgentRequest', request_type_name='DialogflowProjectsLocationsAgentTrainRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def UpdateFulfillment(self, request, global_params=None):
        """Updates the fulfillment.

      Args:
        request: (DialogflowProjectsLocationsAgentUpdateFulfillmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Fulfillment) The response message.
      """
        config = self.GetMethodConfig('UpdateFulfillment')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateFulfillment.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/agent/fulfillment', http_method='PATCH', method_id='dialogflow.projects.locations.agent.updateFulfillment', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Fulfillment', request_type_name='DialogflowProjectsLocationsAgentUpdateFulfillmentRequest', response_type_name='GoogleCloudDialogflowV2Fulfillment', supports_download=False)