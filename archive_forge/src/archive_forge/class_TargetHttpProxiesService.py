from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetHttpProxiesService(base_api.BaseApiService):
    """Service class for the targetHttpProxies resource."""
    _NAME = 'targetHttpProxies'

    def __init__(self, client):
        super(ComputeBeta.TargetHttpProxiesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all TargetHttpProxy resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeTargetHttpProxiesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpProxyAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetHttpProxies.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/targetHttpProxies', request_field='', request_type_name='ComputeTargetHttpProxiesAggregatedListRequest', response_type_name='TargetHttpProxyAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified TargetHttpProxy resource.

      Args:
        request: (ComputeTargetHttpProxiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetHttpProxies.delete', ordered_params=['project', 'targetHttpProxy'], path_params=['project', 'targetHttpProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetHttpProxies/{targetHttpProxy}', request_field='', request_type_name='ComputeTargetHttpProxiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified TargetHttpProxy resource.

      Args:
        request: (ComputeTargetHttpProxiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpProxy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetHttpProxies.get', ordered_params=['project', 'targetHttpProxy'], path_params=['project', 'targetHttpProxy'], query_params=[], relative_path='projects/{project}/global/targetHttpProxies/{targetHttpProxy}', request_field='', request_type_name='ComputeTargetHttpProxiesGetRequest', response_type_name='TargetHttpProxy', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a TargetHttpProxy resource in the specified project using the data included in the request.

      Args:
        request: (ComputeTargetHttpProxiesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetHttpProxies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/targetHttpProxies', request_field='targetHttpProxy', request_type_name='ComputeTargetHttpProxiesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of TargetHttpProxy resources available to the specified project.

      Args:
        request: (ComputeTargetHttpProxiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetHttpProxyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetHttpProxies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/targetHttpProxies', request_field='', request_type_name='ComputeTargetHttpProxiesListRequest', response_type_name='TargetHttpProxyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified TargetHttpProxy resource with the data included in the request. This method supports PATCH semantics and uses JSON merge patch format and processing rules.

      Args:
        request: (ComputeTargetHttpProxiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.targetHttpProxies.patch', ordered_params=['project', 'targetHttpProxy'], path_params=['project', 'targetHttpProxy'], query_params=['requestId'], relative_path='projects/{project}/global/targetHttpProxies/{targetHttpProxy}', request_field='targetHttpProxyResource', request_type_name='ComputeTargetHttpProxiesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetUrlMap(self, request, global_params=None):
        """Changes the URL map for TargetHttpProxy.

      Args:
        request: (ComputeTargetHttpProxiesSetUrlMapRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetUrlMap')
        return self._RunMethod(config, request, global_params=global_params)
    SetUrlMap.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetHttpProxies.setUrlMap', ordered_params=['project', 'targetHttpProxy'], path_params=['project', 'targetHttpProxy'], query_params=['requestId'], relative_path='projects/{project}/targetHttpProxies/{targetHttpProxy}/setUrlMap', request_field='urlMapReference', request_type_name='ComputeTargetHttpProxiesSetUrlMapRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetHttpProxiesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetHttpProxies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/targetHttpProxies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetHttpProxiesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)