import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class BucketsService(base_api.BaseApiService):
    """Service class for the buckets resource."""
    _NAME = u'buckets'

    def __init__(self, client):
        super(StorageV1.BucketsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Permanently deletes an empty bucket.

      Args:
        request: (StorageBucketsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageBucketsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'storage.buckets.delete', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'userProject'], relative_path=u'b/{bucket}', request_field='', request_type_name=u'StorageBucketsDeleteRequest', response_type_name=u'StorageBucketsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata for the specified bucket.

      Args:
        request: (StorageBucketsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Bucket) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.buckets.get', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'projection', u'userProject'], relative_path=u'b/{bucket}', request_field='', request_type_name=u'StorageBucketsGetRequest', response_type_name=u'Bucket', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Returns an IAM policy for the specified bucket.

      Args:
        request: (StorageBucketsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.buckets.getIamPolicy', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'provisionalUserProject', u'optionsRequestedPolicyVersion', u'userProject'], relative_path=u'b/{bucket}/iam', request_field='', request_type_name=u'StorageBucketsGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new bucket.

      Args:
        request: (StorageBucketsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Bucket) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.buckets.insert', ordered_params=[u'project'], path_params=[], query_params=[u'predefinedAcl', u'predefinedDefaultObjectAcl', u'project', u'projection', u'userProject'], relative_path=u'b', request_field=u'bucket', request_type_name=u'StorageBucketsInsertRequest', response_type_name=u'Bucket', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of buckets for a given project.

      Args:
        request: (StorageBucketsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Buckets) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.buckets.list', ordered_params=[u'project'], path_params=[], query_params=[u'maxResults', u'pageToken', u'prefix', u'project', u'projection', u'userProject'], relative_path=u'b', request_field='', request_type_name=u'StorageBucketsListRequest', response_type_name=u'Buckets', supports_download=False)

    def ListChannels(self, request, global_params=None):
        """List active object change notification channels for this bucket.

      Args:
        request: (StorageBucketsListChannelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Channels) The response message.
      """
        config = self.GetMethodConfig('ListChannels')
        return self._RunMethod(config, request, global_params=global_params)
    ListChannels.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.buckets.listChannels', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'userProject'], relative_path=u'b/{bucket}/channels', request_field='', request_type_name=u'StorageBucketsListChannelsRequest', response_type_name=u'Channels', supports_download=False)

    def LockRetentionPolicy(self, request, global_params=None):
        """Locks retention policy on a bucket.

      Args:
        request: (StorageBucketsLockRetentionPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Bucket) The response message.
      """
        config = self.GetMethodConfig('LockRetentionPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    LockRetentionPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.buckets.lockRetentionPolicy', ordered_params=[u'bucket', u'ifMetagenerationMatch'], path_params=[u'bucket'], query_params=[u'ifMetagenerationMatch', u'userProject'], relative_path=u'b/{bucket}/lockRetentionPolicy', request_field='', request_type_name=u'StorageBucketsLockRetentionPolicyRequest', response_type_name=u'Bucket', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches a bucket. Changes to the bucket will be readable immediately after writing, but configuration changes may take time to propagate.

      Args:
        request: (StorageBucketsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Bucket) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'storage.buckets.patch', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'predefinedAcl', u'predefinedDefaultObjectAcl', u'projection', u'userProject'], relative_path=u'b/{bucket}', request_field=u'bucketResource', request_type_name=u'StorageBucketsPatchRequest', response_type_name=u'Bucket', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Updates an IAM policy for the specified bucket.

      Args:
        request: (StorageBucketsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.buckets.setIamPolicy', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'userProject'], relative_path=u'b/{bucket}/iam', request_field=u'policy', request_type_name=u'StorageBucketsSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests a set of permissions on the given bucket to see which, if any, are held by the caller.

      Args:
        request: (StorageBucketsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.buckets.testIamPermissions', ordered_params=[u'bucket', u'permissions'], path_params=[u'bucket'], query_params=[u'permissions', u'userProject'], relative_path=u'b/{bucket}/iam/testPermissions', request_field='', request_type_name=u'StorageBucketsTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a bucket. Changes to the bucket will be readable immediately after writing, but configuration changes may take time to propagate.

      Args:
        request: (StorageBucketsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Bucket) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.buckets.update', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'predefinedAcl', u'predefinedDefaultObjectAcl', u'projection', u'userProject'], relative_path=u'b/{bucket}', request_field=u'bucketResource', request_type_name=u'StorageBucketsUpdateRequest', response_type_name=u'Bucket', supports_download=False)