from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class BackendBucketsService(base_api.BaseApiService):
    """Service class for the backendBuckets resource."""
    _NAME = 'backendBuckets'

    def __init__(self, client):
        super(ComputeBeta.BackendBucketsService, self).__init__(client)
        self._upload_configs = {}

    def AddSignedUrlKey(self, request, global_params=None):
        """Adds a key for validating requests with signed URLs for this backend bucket.

      Args:
        request: (ComputeBackendBucketsAddSignedUrlKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddSignedUrlKey')
        return self._RunMethod(config, request, global_params=global_params)
    AddSignedUrlKey.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendBuckets.addSignedUrlKey', ordered_params=['project', 'backendBucket'], path_params=['backendBucket', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendBuckets/{backendBucket}/addSignedUrlKey', request_field='signedUrlKey', request_type_name='ComputeBackendBucketsAddSignedUrlKeyRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified BackendBucket resource.

      Args:
        request: (ComputeBackendBucketsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.backendBuckets.delete', ordered_params=['project', 'backendBucket'], path_params=['backendBucket', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendBuckets/{backendBucket}', request_field='', request_type_name='ComputeBackendBucketsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteSignedUrlKey(self, request, global_params=None):
        """Deletes a key for validating requests with signed URLs for this backend bucket.

      Args:
        request: (ComputeBackendBucketsDeleteSignedUrlKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DeleteSignedUrlKey')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteSignedUrlKey.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendBuckets.deleteSignedUrlKey', ordered_params=['project', 'backendBucket', 'keyName'], path_params=['backendBucket', 'project'], query_params=['keyName', 'requestId'], relative_path='projects/{project}/global/backendBuckets/{backendBucket}/deleteSignedUrlKey', request_field='', request_type_name='ComputeBackendBucketsDeleteSignedUrlKeyRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified BackendBucket resource.

      Args:
        request: (ComputeBackendBucketsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendBucket) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendBuckets.get', ordered_params=['project', 'backendBucket'], path_params=['backendBucket', 'project'], query_params=[], relative_path='projects/{project}/global/backendBuckets/{backendBucket}', request_field='', request_type_name='ComputeBackendBucketsGetRequest', response_type_name='BackendBucket', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeBackendBucketsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendBuckets.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/backendBuckets/{resource}/getIamPolicy', request_field='', request_type_name='ComputeBackendBucketsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a BackendBucket resource in the specified project using the data included in the request.

      Args:
        request: (ComputeBackendBucketsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendBuckets.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/backendBuckets', request_field='backendBucket', request_type_name='ComputeBackendBucketsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of BackendBucket resources available to the specified project.

      Args:
        request: (ComputeBackendBucketsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackendBucketList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.backendBuckets.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/backendBuckets', request_field='', request_type_name='ComputeBackendBucketsListRequest', response_type_name='BackendBucketList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified BackendBucket resource with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeBackendBucketsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.backendBuckets.patch', ordered_params=['project', 'backendBucket'], path_params=['backendBucket', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendBuckets/{backendBucket}', request_field='backendBucketResource', request_type_name='ComputeBackendBucketsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetEdgeSecurityPolicy(self, request, global_params=None):
        """Sets the edge security policy for the specified backend bucket.

      Args:
        request: (ComputeBackendBucketsSetEdgeSecurityPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetEdgeSecurityPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetEdgeSecurityPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendBuckets.setEdgeSecurityPolicy', ordered_params=['project', 'backendBucket'], path_params=['backendBucket', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendBuckets/{backendBucket}/setEdgeSecurityPolicy', request_field='securityPolicyReference', request_type_name='ComputeBackendBucketsSetEdgeSecurityPolicyRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeBackendBucketsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendBuckets.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/backendBuckets/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeBackendBucketsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeBackendBucketsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.backendBuckets.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/backendBuckets/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeBackendBucketsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified BackendBucket resource with the data included in the request.

      Args:
        request: (ComputeBackendBucketsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.backendBuckets.update', ordered_params=['project', 'backendBucket'], path_params=['backendBucket', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/backendBuckets/{backendBucket}', request_field='backendBucketResource', request_type_name='ComputeBackendBucketsUpdateRequest', response_type_name='Operation', supports_download=False)