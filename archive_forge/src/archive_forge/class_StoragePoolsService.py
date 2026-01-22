from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class StoragePoolsService(base_api.BaseApiService):
    """Service class for the storagePools resource."""
    _NAME = 'storagePools'

    def __init__(self, client):
        super(ComputeBeta.StoragePoolsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of storage pools. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeStoragePoolsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePools.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/storagePools', request_field='', request_type_name='ComputeStoragePoolsAggregatedListRequest', response_type_name='StoragePoolAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified storage pool. Deleting a storagePool removes its data permanently and is irreversible. However, deleting a storagePool does not delete any snapshots previously made from the storagePool. You must separately delete snapshots.

      Args:
        request: (ComputeStoragePoolsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.storagePools.delete', ordered_params=['project', 'zone', 'storagePool'], path_params=['project', 'storagePool', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/storagePools/{storagePool}', request_field='', request_type_name='ComputeStoragePoolsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a specified storage pool. Gets a list of available storage pools by making a list() request.

      Args:
        request: (ComputeStoragePoolsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePool) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePools.get', ordered_params=['project', 'zone', 'storagePool'], path_params=['project', 'storagePool', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/storagePools/{storagePool}', request_field='', request_type_name='ComputeStoragePoolsGetRequest', response_type_name='StoragePool', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeStoragePoolsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePools.getIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/zones/{zone}/storagePools/{resource}/getIamPolicy', request_field='', request_type_name='ComputeStoragePoolsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a storage pool in the specified project using the data in the request.

      Args:
        request: (ComputeStoragePoolsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.storagePools.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/storagePools', request_field='storagePool', request_type_name='ComputeStoragePoolsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of storage pools contained within the specified zone.

      Args:
        request: (ComputeStoragePoolsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePools.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/storagePools', request_field='', request_type_name='ComputeStoragePoolsListRequest', response_type_name='StoragePoolList', supports_download=False)

    def ListDisks(self, request, global_params=None):
        """Lists the disks in a specified storage pool.

      Args:
        request: (ComputeStoragePoolsListDisksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolListDisks) The response message.
      """
        config = self.GetMethodConfig('ListDisks')
        return self._RunMethod(config, request, global_params=global_params)
    ListDisks.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.storagePools.listDisks', ordered_params=['project', 'zone', 'storagePool'], path_params=['project', 'storagePool', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/storagePools/{storagePool}/listDisks', request_field='', request_type_name='ComputeStoragePoolsListDisksRequest', response_type_name='StoragePoolListDisks', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeStoragePoolsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.storagePools.setIamPolicy', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/storagePools/{resource}/setIamPolicy', request_field='zoneSetPolicyRequest', request_type_name='ComputeStoragePoolsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeStoragePoolsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.storagePools.testIamPermissions', ordered_params=['project', 'zone', 'resource'], path_params=['project', 'resource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/storagePools/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeStoragePoolsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified storagePool with the data included in the request. The update is performed only on selected fields included as part of update-mask. Only the following fields can be modified: size_tb and provisioned_iops.

      Args:
        request: (ComputeStoragePoolsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.storagePools.update', ordered_params=['project', 'zone', 'storagePool'], path_params=['project', 'storagePool', 'zone'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/zones/{zone}/storagePools/{storagePool}', request_field='storagePoolResource', request_type_name='ComputeStoragePoolsUpdateRequest', response_type_name='Operation', supports_download=False)