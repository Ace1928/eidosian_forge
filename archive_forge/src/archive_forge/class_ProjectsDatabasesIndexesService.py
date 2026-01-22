from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1beta1 import firestore_v1beta1_messages as messages
class ProjectsDatabasesIndexesService(base_api.BaseApiService):
    """Service class for the projects_databases_indexes resource."""
    _NAME = 'projects_databases_indexes'

    def __init__(self, client):
        super(FirestoreV1beta1.ProjectsDatabasesIndexesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates the specified index. A newly created index's initial state is `CREATING`. On completion of the returned google.longrunning.Operation, the state will be `READY`. If the index already exists, the call will return an `ALREADY_EXISTS` status. During creation, the process could result in an error, in which case the index will move to the `ERROR` state. The process can be recovered by fixing the data that caused the error, removing the index with delete, then re-creating the index with create. Indexes with a single field cannot be created.

      Args:
        request: (FirestoreProjectsDatabasesIndexesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/databases/{databasesId}/indexes', http_method='POST', method_id='firestore.projects.databases.indexes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/indexes', request_field='googleFirestoreAdminV1beta1Index', request_type_name='FirestoreProjectsDatabasesIndexesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an index.

      Args:
        request: (FirestoreProjectsDatabasesIndexesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/databases/{databasesId}/indexes/{indexesId}', http_method='DELETE', method_id='firestore.projects.databases.indexes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesIndexesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an index.

      Args:
        request: (FirestoreProjectsDatabasesIndexesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1beta1Index) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/databases/{databasesId}/indexes/{indexesId}', http_method='GET', method_id='firestore.projects.databases.indexes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesIndexesGetRequest', response_type_name='GoogleFirestoreAdminV1beta1Index', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the indexes that match the specified filters.

      Args:
        request: (FirestoreProjectsDatabasesIndexesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1beta1ListIndexesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/databases/{databasesId}/indexes', http_method='GET', method_id='firestore.projects.databases.indexes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/indexes', request_field='', request_type_name='FirestoreProjectsDatabasesIndexesListRequest', response_type_name='GoogleFirestoreAdminV1beta1ListIndexesResponse', supports_download=False)