from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsAgentKnowledgeBasesDocumentsService(base_api.BaseApiService):
    """Service class for the projects_agent_knowledgeBases_documents resource."""
    _NAME = 'projects_agent_knowledgeBases_documents'

    def __init__(self, client):
        super(DialogflowV2.ProjectsAgentKnowledgeBasesDocumentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new document. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDocumentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}/documents', http_method='POST', method_id='dialogflow.projects.agent.knowledgeBases.documents.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/documents', request_field='googleCloudDialogflowV2Document', request_type_name='DialogflowProjectsAgentKnowledgeBasesDocumentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified document. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty).

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDocumentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}', http_method='DELETE', method_id='dialogflow.projects.agent.knowledgeBases.documents.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentKnowledgeBasesDocumentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified document.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDocumentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Document) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}', http_method='GET', method_id='dialogflow.projects.agent.knowledgeBases.documents.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsAgentKnowledgeBasesDocumentsGetRequest', response_type_name='GoogleCloudDialogflowV2Document', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all documents of the knowledge base.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDocumentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListDocumentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}/documents', http_method='GET', method_id='dialogflow.projects.agent.knowledgeBases.documents.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/documents', request_field='', request_type_name='DialogflowProjectsAgentKnowledgeBasesDocumentsListRequest', response_type_name='GoogleCloudDialogflowV2ListDocumentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified document. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDocumentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}', http_method='PATCH', method_id='dialogflow.projects.agent.knowledgeBases.documents.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Document', request_type_name='DialogflowProjectsAgentKnowledgeBasesDocumentsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Reload(self, request, global_params=None):
        """Reloads the specified document from its specified source, content_uri or content. The previously loaded content of the document will be deleted. Note: Even when the content of the document has not changed, there still may be side effects because of internal implementation changes. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document Note: The `projects.agent.knowledgeBases.documents` resource is deprecated; only use `projects.knowledgeBases.documents`.

      Args:
        request: (DialogflowProjectsAgentKnowledgeBasesDocumentsReloadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Reload')
        return self._RunMethod(config, request, global_params=global_params)
    Reload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/agent/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}:reload', http_method='POST', method_id='dialogflow.projects.agent.knowledgeBases.documents.reload', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:reload', request_field='googleCloudDialogflowV2ReloadDocumentRequest', request_type_name='DialogflowProjectsAgentKnowledgeBasesDocumentsReloadRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)