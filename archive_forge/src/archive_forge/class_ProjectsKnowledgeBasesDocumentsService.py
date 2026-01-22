from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dialogflow.v2 import dialogflow_v2_messages as messages
class ProjectsKnowledgeBasesDocumentsService(base_api.BaseApiService):
    """Service class for the projects_knowledgeBases_documents resource."""
    _NAME = 'projects_knowledgeBases_documents'

    def __init__(self, client):
        super(DialogflowV2.ProjectsKnowledgeBasesDocumentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new document. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents', http_method='POST', method_id='dialogflow.projects.knowledgeBases.documents.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/documents', request_field='googleCloudDialogflowV2Document', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified document. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: An [Empty message](https://developers.google.com/protocol-buffers/docs/reference/google.protobuf#empty).

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}', http_method='DELETE', method_id='dialogflow.projects.knowledgeBases.documents.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports a smart messaging candidate document into the specified destination. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}:export', http_method='POST', method_id='dialogflow.projects.knowledgeBases.documents.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:export', request_field='googleCloudDialogflowV2ExportDocumentRequest', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsExportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves the specified document.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2Document) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}', http_method='GET', method_id='dialogflow.projects.knowledgeBases.documents.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsGetRequest', response_type_name='GoogleCloudDialogflowV2Document', supports_download=False)

    def Import(self, request, global_params=None):
        """Creates documents by importing data from external sources. Dialogflow supports up to 350 documents in each request. If you try to import more, Dialogflow will return an error. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: ImportDocumentsResponse.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents:import', http_method='POST', method_id='dialogflow.projects.knowledgeBases.documents.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/documents:import', request_field='googleCloudDialogflowV2ImportDocumentsRequest', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of all documents of the knowledge base.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDialogflowV2ListDocumentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents', http_method='GET', method_id='dialogflow.projects.knowledgeBases.documents.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/documents', request_field='', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsListRequest', response_type_name='GoogleCloudDialogflowV2ListDocumentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified document. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}', http_method='PATCH', method_id='dialogflow.projects.knowledgeBases.documents.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudDialogflowV2Document', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Reload(self, request, global_params=None):
        """Reloads the specified document from its specified source, content_uri or content. The previously loaded content of the document will be deleted. Note: Even when the content of the document has not changed, there still may be side effects because of internal implementation changes. This method is a [long-running operation](https://cloud.google.com/dialogflow/cx/docs/how/long-running-operation). The returned `Operation` type has the following method-specific fields: - `metadata`: KnowledgeOperationMetadata - `response`: Document Note: The `projects.agent.knowledgeBases.documents` resource is deprecated; only use `projects.knowledgeBases.documents`.

      Args:
        request: (DialogflowProjectsKnowledgeBasesDocumentsReloadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Reload')
        return self._RunMethod(config, request, global_params=global_params)
    Reload.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/knowledgeBases/{knowledgeBasesId}/documents/{documentsId}:reload', http_method='POST', method_id='dialogflow.projects.knowledgeBases.documents.reload', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:reload', request_field='googleCloudDialogflowV2ReloadDocumentRequest', request_type_name='DialogflowProjectsKnowledgeBasesDocumentsReloadRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)