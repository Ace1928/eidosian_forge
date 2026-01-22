from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsub.v1 import pubsub_v1_messages as messages
class ProjectsSchemasService(base_api.BaseApiService):
    """Service class for the projects_schemas resource."""
    _NAME = 'projects_schemas'

    def __init__(self, client):
        super(PubsubV1.ProjectsSchemasService, self).__init__(client)
        self._upload_configs = {}

    def Commit(self, request, global_params=None):
        """Commits a new schema revision to an existing schema.

      Args:
        request: (PubsubProjectsSchemasCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:commit', http_method='POST', method_id='pubsub.projects.schemas.commit', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:commit', request_field='commitSchemaRequest', request_type_name='PubsubProjectsSchemasCommitRequest', response_type_name='Schema', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a schema.

      Args:
        request: (PubsubProjectsSchemasCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas', http_method='POST', method_id='pubsub.projects.schemas.create', ordered_params=['parent'], path_params=['parent'], query_params=['schemaId'], relative_path='v1/{+parent}/schemas', request_field='schema', request_type_name='PubsubProjectsSchemasCreateRequest', response_type_name='Schema', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a schema.

      Args:
        request: (PubsubProjectsSchemasDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}', http_method='DELETE', method_id='pubsub.projects.schemas.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='PubsubProjectsSchemasDeleteRequest', response_type_name='Empty', supports_download=False)

    def DeleteRevision(self, request, global_params=None):
        """Deletes a specific schema revision.

      Args:
        request: (PubsubProjectsSchemasDeleteRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('DeleteRevision')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:deleteRevision', http_method='DELETE', method_id='pubsub.projects.schemas.deleteRevision', ordered_params=['name'], path_params=['name'], query_params=['revisionId'], relative_path='v1/{+name}:deleteRevision', request_field='', request_type_name='PubsubProjectsSchemasDeleteRevisionRequest', response_type_name='Schema', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a schema.

      Args:
        request: (PubsubProjectsSchemasGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}', http_method='GET', method_id='pubsub.projects.schemas.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='PubsubProjectsSchemasGetRequest', response_type_name='Schema', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (PubsubProjectsSchemasGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:getIamPolicy', http_method='GET', method_id='pubsub.projects.schemas.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='PubsubProjectsSchemasGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists schemas in a project.

      Args:
        request: (PubsubProjectsSchemasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSchemasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas', http_method='GET', method_id='pubsub.projects.schemas.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/schemas', request_field='', request_type_name='PubsubProjectsSchemasListRequest', response_type_name='ListSchemasResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """Lists all schema revisions for the named schema.

      Args:
        request: (PubsubProjectsSchemasListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSchemaRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:listRevisions', http_method='GET', method_id='pubsub.projects.schemas.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+name}:listRevisions', request_field='', request_type_name='PubsubProjectsSchemasListRevisionsRequest', response_type_name='ListSchemaRevisionsResponse', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Creates a new schema revision that is a copy of the provided revision_id.

      Args:
        request: (PubsubProjectsSchemasRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:rollback', http_method='POST', method_id='pubsub.projects.schemas.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='rollbackSchemaRequest', request_type_name='PubsubProjectsSchemasRollbackRequest', response_type_name='Schema', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (PubsubProjectsSchemasSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:setIamPolicy', http_method='POST', method_id='pubsub.projects.schemas.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='PubsubProjectsSchemasSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (PubsubProjectsSchemasTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas/{schemasId}:testIamPermissions', http_method='POST', method_id='pubsub.projects.schemas.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='PubsubProjectsSchemasTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Validate(self, request, global_params=None):
        """Validates a schema.

      Args:
        request: (PubsubProjectsSchemasValidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateSchemaResponse) The response message.
      """
        config = self.GetMethodConfig('Validate')
        return self._RunMethod(config, request, global_params=global_params)
    Validate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas:validate', http_method='POST', method_id='pubsub.projects.schemas.validate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/schemas:validate', request_field='validateSchemaRequest', request_type_name='PubsubProjectsSchemasValidateRequest', response_type_name='ValidateSchemaResponse', supports_download=False)

    def ValidateMessage(self, request, global_params=None):
        """Validates a message against a schema.

      Args:
        request: (PubsubProjectsSchemasValidateMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ValidateMessageResponse) The response message.
      """
        config = self.GetMethodConfig('ValidateMessage')
        return self._RunMethod(config, request, global_params=global_params)
    ValidateMessage.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/schemas:validateMessage', http_method='POST', method_id='pubsub.projects.schemas.validateMessage', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/schemas:validateMessage', request_field='validateMessageRequest', request_type_name='PubsubProjectsSchemasValidateMessageRequest', response_type_name='ValidateMessageResponse', supports_download=False)