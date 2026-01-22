from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1alpha1 import containeranalysis_v1alpha1_messages as messages
class ProvidersNotesService(base_api.BaseApiService):
    """Service class for the providers_notes resource."""
    _NAME = 'providers_notes'

    def __init__(self, client):
        super(ContaineranalysisV1alpha1.ProvidersNotesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `Note`.

      Args:
        request: (ContaineranalysisProvidersNotesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes', http_method='POST', method_id='containeranalysis.providers.notes.create', ordered_params=['name'], path_params=['name'], query_params=['noteId', 'parent'], relative_path='v1alpha1/{+name}/notes', request_field='note', request_type_name='ContaineranalysisProvidersNotesCreateRequest', response_type_name='Note', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the given `Note` from the system.

      Args:
        request: (ContaineranalysisProvidersNotesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}', http_method='DELETE', method_id='containeranalysis.providers.notes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='ContaineranalysisProvidersNotesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested `Note`.

      Args:
        request: (ContaineranalysisProvidersNotesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}', http_method='GET', method_id='containeranalysis.providers.notes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='ContaineranalysisProvidersNotesGetRequest', response_type_name='Note', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a note or an `Occurrence` resource. Requires `containeranalysis.notes.setIamPolicy` or `containeranalysis.occurrences.setIamPolicy` permission if the resource is a note or occurrence, respectively. Attempting to call this method on a resource without the required permission will result in a `PERMISSION_DENIED` error. Attempting to call this method on a non-existent resource will result in a `NOT_FOUND` error if the user has list permission on the project, or a `PERMISSION_DENIED` error otherwise. The resource takes the following formats: `projects/{PROJECT_ID}/occurrences/{OCCURRENCE_ID}` for occurrences and projects/{PROJECT_ID}/notes/{NOTE_ID} for notes.

      Args:
        request: (ContaineranalysisProvidersNotesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}:getIamPolicy', http_method='POST', method_id='containeranalysis.providers.notes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='ContaineranalysisProvidersNotesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all `Notes` for a given project.

      Args:
        request: (ContaineranalysisProvidersNotesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNotesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes', http_method='GET', method_id='containeranalysis.providers.notes.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken', 'parent'], relative_path='v1alpha1/{+name}/notes', request_field='', request_type_name='ContaineranalysisProvidersNotesListRequest', response_type_name='ListNotesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing `Note`.

      Args:
        request: (ContaineranalysisProvidersNotesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}', http_method='PATCH', method_id='containeranalysis.providers.notes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='note', request_type_name='ContaineranalysisProvidersNotesPatchRequest', response_type_name='Note', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified `Note` or `Occurrence`. Requires `containeranalysis.notes.setIamPolicy` or `containeranalysis.occurrences.setIamPolicy` permission if the resource is a `Note` or an `Occurrence`, respectively. Attempting to call this method without these permissions will result in a ` `PERMISSION_DENIED` error. Attempting to call this method on a non-existent resource will result in a `NOT_FOUND` error if the user has `containeranalysis.notes.list` permission on a `Note` or `containeranalysis.occurrences.list` on an `Occurrence`, or a `PERMISSION_DENIED` error otherwise. The resource takes the following formats: `projects/{projectid}/occurrences/{occurrenceid}` for occurrences and projects/{projectid}/notes/{noteid} for notes.

      Args:
        request: (ContaineranalysisProvidersNotesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}:setIamPolicy', http_method='POST', method_id='containeranalysis.providers.notes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ContaineranalysisProvidersNotesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the permissions that a caller has on the specified note or occurrence resource. Requires list permission on the project (for example, "storage.objects.list" on the containing bucket for testing permission of an object). Attempting to call this method on a non-existent resource will result in a `NOT_FOUND` error if the user has list permission on the project, or a `PERMISSION_DENIED` error otherwise. The resource takes the following formats: `projects/{PROJECT_ID}/occurrences/{OCCURRENCE_ID}` for `Occurrences` and `projects/{PROJECT_ID}/notes/{NOTE_ID}` for `Notes`.

      Args:
        request: (ContaineranalysisProvidersNotesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/providers/{providersId}/notes/{notesId}:testIamPermissions', http_method='POST', method_id='containeranalysis.providers.notes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ContaineranalysisProvidersNotesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)