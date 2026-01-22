from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class UrlMapsService(base_api.BaseApiService):
    """Service class for the urlMaps resource."""
    _NAME = 'urlMaps'

    def __init__(self, client):
        super(ComputeBeta.UrlMapsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all UrlMap resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeUrlMapsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMapsAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.urlMaps.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/urlMaps', request_field='', request_type_name='ComputeUrlMapsAggregatedListRequest', response_type_name='UrlMapsAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified UrlMap resource.

      Args:
        request: (ComputeUrlMapsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.urlMaps.delete', ordered_params=['project', 'urlMap'], path_params=['project', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/global/urlMaps/{urlMap}', request_field='', request_type_name='ComputeUrlMapsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified UrlMap resource.

      Args:
        request: (ComputeUrlMapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.urlMaps.get', ordered_params=['project', 'urlMap'], path_params=['project', 'urlMap'], query_params=[], relative_path='projects/{project}/global/urlMaps/{urlMap}', request_field='', request_type_name='ComputeUrlMapsGetRequest', response_type_name='UrlMap', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a UrlMap resource in the specified project using the data included in the request.

      Args:
        request: (ComputeUrlMapsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.urlMaps.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/urlMaps', request_field='urlMap', request_type_name='ComputeUrlMapsInsertRequest', response_type_name='Operation', supports_download=False)

    def InvalidateCache(self, request, global_params=None):
        """Initiates a cache invalidation operation, invalidating the specified path, scoped to the specified UrlMap. For more information, see [Invalidating cached content](/cdn/docs/invalidating-cached-content).

      Args:
        request: (ComputeUrlMapsInvalidateCacheRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('InvalidateCache')
        return self._RunMethod(config, request, global_params=global_params)
    InvalidateCache.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.urlMaps.invalidateCache', ordered_params=['project', 'urlMap'], path_params=['project', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/global/urlMaps/{urlMap}/invalidateCache', request_field='cacheInvalidationRule', request_type_name='ComputeUrlMapsInvalidateCacheRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of UrlMap resources available to the specified project.

      Args:
        request: (ComputeUrlMapsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMapList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.urlMaps.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/urlMaps', request_field='', request_type_name='ComputeUrlMapsListRequest', response_type_name='UrlMapList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified UrlMap resource with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules.

      Args:
        request: (ComputeUrlMapsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.urlMaps.patch', ordered_params=['project', 'urlMap'], path_params=['project', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/global/urlMaps/{urlMap}', request_field='urlMapResource', request_type_name='ComputeUrlMapsPatchRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeUrlMapsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.urlMaps.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/urlMaps/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeUrlMapsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified UrlMap resource with the data included in the request.

      Args:
        request: (ComputeUrlMapsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='compute.urlMaps.update', ordered_params=['project', 'urlMap'], path_params=['project', 'urlMap'], query_params=['requestId'], relative_path='projects/{project}/global/urlMaps/{urlMap}', request_field='urlMapResource', request_type_name='ComputeUrlMapsUpdateRequest', response_type_name='Operation', supports_download=False)

    def Validate(self, request, global_params=None):
        """Runs static validation for the UrlMap. In particular, the tests of the provided UrlMap will be run. Calling this method does NOT create the UrlMap.

      Args:
        request: (ComputeUrlMapsValidateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UrlMapsValidateResponse) The response message.
      """
        config = self.GetMethodConfig('Validate')
        return self._RunMethod(config, request, global_params=global_params)
    Validate.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.urlMaps.validate', ordered_params=['project', 'urlMap'], path_params=['project', 'urlMap'], query_params=[], relative_path='projects/{project}/global/urlMaps/{urlMap}/validate', request_field='urlMapsValidateRequest', request_type_name='ComputeUrlMapsValidateRequest', response_type_name='UrlMapsValidateResponse', supports_download=False)