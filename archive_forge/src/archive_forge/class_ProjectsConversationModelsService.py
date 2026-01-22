from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsConversationModelsService(base_api.BaseApiService):
    """Service class for the projects_conversationModels resource."""
    _NAME = 'projects_conversationModels'

    def __init__(self, client):
        super(DialogflowV2.ProjectsConversationModelsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a model. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: CreateConversationModelOperationMetadata - `response`: ConversationModel.

      Args:
        request: (DialogflowProjectsConversationModelsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels', http_method='POST', method_id='dialogflow.projects.conversationModels.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/conversationModels', request_field='googleCloudDialogflowV2ConversationModel', request_type_name='DialogflowProjectsConversationModelsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a model. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: DeleteConversationModelOperationMetadata - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty).

      Args:
        request: (DialogflowProjectsConversationModelsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels/{conversationModelsId}', http_method='DELETE', method_id='dialogflow.projects.conversationModels.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsConversationModelsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Deploy(self, request, global_params=None):
        """Deploys a model. If a model is already deployed, deploying it has no effect. A model can only serve prediction requests after it gets deployed. For article suggestion, custom model will not be used unless it is deployed. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: DeployConversationModelOperationMetadata - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty).

      Args:
        request: (DialogflowProjectsConversationModelsDeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Deploy')
        return self._RunMethod(config, request, global_params=global_params)
    Deploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels/{conversationModelsId}:deploy', http_method='POST', method_id='dialogflow.projects.conversationModels.deploy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:deploy', request_field='googleCloudDialogflowV2DeployConversationModelRequest', request_type_name='DialogflowProjectsConversationModelsDeployRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets conversation model.

      Args:
        request: (DialogflowProjectsConversationModelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ConversationModel) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels/{conversationModelsId}', http_method='GET', method_id='dialogflow.projects.conversationModels.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsConversationModelsGetRequest', response_type_name='GoogleCloudDialogflowV2ConversationModel', supports_download=False)

    def List(self, request, global_params=None):
        """Lists conversation models.

      Args:
        request: (DialogflowProjectsConversationModelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListConversationModelsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels', http_method='GET', method_id='dialogflow.projects.conversationModels.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/conversationModels', request_field='', request_type_name='DialogflowProjectsConversationModelsListRequest', response_type_name='GoogleCloudDialogflowV2ListConversationModelsResponse', supports_download=False)

    def Undeploy(self, request, global_params=None):
        """Undeploys a model. If the model is not deployed this method has no effect. If the model is currently being used: - For article suggestion, article suggestion will fallback to the default model if model is undeployed. This method is a [long-running operation](https://cloud.google.com/dialogflow/es/docs/how/long-running-operations). The returned `Operation` type has the following method-specific fields: - `metadata`: UndeployConversationModelOperationMetadata - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty).

      Args:
        request: (DialogflowProjectsConversationModelsUndeployRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Undeploy')
        return self._RunMethod(config, request, global_params=global_params)
    Undeploy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/conversationModels/{conversationModelsId}:undeploy', http_method='POST', method_id='dialogflow.projects.conversationModels.undeploy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undeploy', request_field='googleCloudDialogflowV2UndeployConversationModelRequest', request_type_name='DialogflowProjectsConversationModelsUndeployRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)