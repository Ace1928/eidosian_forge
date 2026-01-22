from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetGrpcProxiesService(base_api.BaseApiService):
    """Service class for the targetGrpcProxies resource."""
    _NAME = 'targetGrpcProxies'

    def __init__(self, client):
        super(ComputeBeta.TargetGrpcProxiesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetGrpcProxy in the given scope.

      Args:
        request: (ComputeTargetGrpcProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetGrpcProxies.delete', ordered_params=['project', 'targetGrpcProxy'], path_params=['project', 'targetGrpcProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetGrpcProxies/{targetGrpcProxy}', request_field='', request_type_name='ComputeTargetGrpcProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetGrpcProxy resource in the given scope.

      Args:
        request: (ComputeTargetGrpcProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetGrpcProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetGrpcProxies.get', ordered_params=['project', 'targetGrpcProxy'], path_params=['project', 'targetGrpcProxy'], query_params=[], relative_path='projects/{project}/global/targetGrpcProxies/{targetGrpcProxy}', request_field='', request_type_name='ComputeTargetGrpcProxiesGetRequest', response_type_name='TargetGrpcProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetGrpcProxy in the specified project in the given scope using the parameters that are included in the request.

      Args:
        request: (ComputeTargetGrpcProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetGrpcProxies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/targetGrpcProxies', request_field='targetGrpcProxy', request_type_name='ComputeTargetGrpcProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the TargetGrpcProxies for a project in the given scope.

      Args:
        request: (ComputeTargetGrpcProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetGrpcProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetGrpcProxies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/targetGrpcProxies', request_field='', request_type_name='ComputeTargetGrpcProxiesListRequest', response_type_name='TargetGrpcProxyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified TargetGrpcProxy resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeTargetGrpcProxiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.targetGrpcProxies.patch', ordered_params=['project', 'targetGrpcProxy'], path_params=['project', 'targetGrpcProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetGrpcProxies/{targetGrpcProxy}', request_field='targetGrpcProxyResource', request_type_name='ComputeTargetGrpcProxiesPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetGrpcProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetGrpcProxies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/targetGrpcProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetGrpcProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)