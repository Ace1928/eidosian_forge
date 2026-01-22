from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsDatabasesCollectionGroupsFieldsService(base_api.BaseApiService):
    """Service class for the projects_databases_collectionGroups_fields resource."""
    _NAME = 'projects_databases_collectionGroups_fields'

    def __init__(self, client):
        super(FirestoreV1.ProjectsDatabasesCollectionGroupsFieldsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the metadata and configuration for a Field.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsFieldsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1Field) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/fields/{fieldsId}', http_method='GET', method_id='firestore.projects.databases.collectionGroups.fields.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesCollectionGroupsFieldsGetRequest', response_type_name='GoogleFirestoreAdminV1Field', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the field configuration and metadata for this database. Currently, FirestoreAdmin.ListFields only supports listing fields that have been explicitly overridden. To issue this query, call FirestoreAdmin.ListFields with the filter set to `indexConfig.usesAncestorConfig:false` or `ttlConfig:*`.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsFieldsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1ListFieldsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/fields', http_method='GET', method_id='firestore.projects.databases.collectionGroups.fields.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/fields', request_field='', request_type_name='FirestoreProjectsDatabasesCollectionGroupsFieldsListRequest', response_type_name='GoogleFirestoreAdminV1ListFieldsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a field configuration. Currently, field updates apply only to single field index configuration. However, calls to FirestoreAdmin.UpdateField should provide a field mask to avoid changing any configuration that the caller isn't aware of. The field mask should be specified as: `{ paths: "index_config" }`. This call returns a google.longrunning.Operation which may be used to track the status of the field update. The metadata for the operation will be the type FieldOperationMetadata. To configure the default field settings for the database, use the special `Field` with resource name: `projects/{project_id}/databases/{database_id}/collectionGroups/__default__/fields/*`.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/fields/{fieldsId}', http_method='PATCH', method_id='firestore.projects.databases.collectionGroups.fields.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleFirestoreAdminV1Field', request_type_name='FirestoreProjectsDatabasesCollectionGroupsFieldsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)