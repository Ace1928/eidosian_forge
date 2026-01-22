from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storage.v1 import storage_v1_messages as messages
class ManagedFoldersService(base_api.BaseApiService):
    """Service class for the managedFolders resource."""
    _NAME = 'managedFolders'

    def __init__(self, client):
        super(StorageV1.ManagedFoldersService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Permanently deletes a managed folder.

      Args:
        request: (StorageManagedFoldersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageManagedFoldersDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='storage.managedFolders.delete', ordered_params=['bucket', 'managedFolder'], path_params=['bucket', 'managedFolder'], query_params=['allowNonEmpty', 'ifMetagenerationMatch', 'ifMetagenerationNotMatch'], relative_path='b/{bucket}/managedFolders/{managedFolder}', request_field='', request_type_name='StorageManagedFoldersDeleteRequest', response_type_name='StorageManagedFoldersDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata of the specified managed folder.

      Args:
        request: (StorageManagedFoldersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedFolder) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.managedFolders.get', ordered_params=['bucket', 'managedFolder'], path_params=['bucket', 'managedFolder'], query_params=['ifMetagenerationMatch', 'ifMetagenerationNotMatch'], relative_path='b/{bucket}/managedFolders/{managedFolder}', request_field='', request_type_name='StorageManagedFoldersGetRequest', response_type_name='ManagedFolder', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Returns an IAM policy for the specified managed folder.

      Args:
        request: (StorageManagedFoldersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.managedFolders.getIamPolicy', ordered_params=['bucket', 'managedFolder'], path_params=['bucket', 'managedFolder'], query_params=['optionsRequestedPolicyVersion', 'userProject'], relative_path='b/{bucket}/managedFolders/{managedFolder}/iam', request_field='', request_type_name='StorageManagedFoldersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new managed folder.

      Args:
        request: (ManagedFolder) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedFolder) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storage.managedFolders.insert', ordered_params=['bucket'], path_params=['bucket'], query_params=[], relative_path='b/{bucket}/managedFolders', request_field='<request>', request_type_name='ManagedFolder', response_type_name='ManagedFolder', supports_download=False)

    def List(self, request, global_params=None):
        """Lists managed folders in the given bucket.

      Args:
        request: (StorageManagedFoldersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedFolders) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.managedFolders.list', ordered_params=['bucket'], path_params=['bucket'], query_params=['pageSize', 'pageToken', 'prefix'], relative_path='b/{bucket}/managedFolders', request_field='', request_type_name='StorageManagedFoldersListRequest', response_type_name='ManagedFolders', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Updates an IAM policy for the specified managed folder.

      Args:
        request: (StorageManagedFoldersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='storage.managedFolders.setIamPolicy', ordered_params=['bucket', 'managedFolder'], path_params=['bucket', 'managedFolder'], query_params=['userProject'], relative_path='b/{bucket}/managedFolders/{managedFolder}/iam', request_field='policy', request_type_name='StorageManagedFoldersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests a set of permissions on the given managed folder to see which, if any, are held by the caller.

      Args:
        request: (StorageManagedFoldersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.managedFolders.testIamPermissions', ordered_params=['bucket', 'managedFolder', 'permissions'], path_params=['bucket', 'managedFolder'], query_params=['permissions', 'userProject'], relative_path='b/{bucket}/managedFolders/{managedFolder}/iam/testPermissions', request_field='', request_type_name='StorageManagedFoldersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)