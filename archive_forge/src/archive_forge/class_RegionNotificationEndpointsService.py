from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionNotificationEndpointsService(base_api.BaseApiService):
    """Service class for the regionNotificationEndpoints resource."""
    _NAME = 'regionNotificationEndpoints'

    def __init__(self, client):
        super(ComputeBeta.RegionNotificationEndpointsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified NotificationEndpoint in the given region.

      Args:
        request: (ComputeRegionNotificationEndpointsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionNotificationEndpoints.delete', ordered_params=['project', 'region', 'notificationEndpoint'], path_params=['notificationEndpoint', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/notificationEndpoints/{notificationEndpoint}', request_field='', request_type_name='ComputeRegionNotificationEndpointsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified NotificationEndpoint resource in the given region.

      Args:
        request: (ComputeRegionNotificationEndpointsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationEndpoint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNotificationEndpoints.get', ordered_params=['project', 'region', 'notificationEndpoint'], path_params=['notificationEndpoint', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/notificationEndpoints/{notificationEndpoint}', request_field='', request_type_name='ComputeRegionNotificationEndpointsGetRequest', response_type_name='NotificationEndpoint', supports_download=False)

    def Insert(self, request, global_params=None):
        """Create a NotificationEndpoint in the specified project in the given region using the parameters that are included in the request.

      Args:
        request: (ComputeRegionNotificationEndpointsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNotificationEndpoints.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/notificationEndpoints', request_field='notificationEndpoint', request_type_name='ComputeRegionNotificationEndpointsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the NotificationEndpoints for a project in the given region.

      Args:
        request: (ComputeRegionNotificationEndpointsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NotificationEndpointList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNotificationEndpoints.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/notificationEndpoints', request_field='', request_type_name='ComputeRegionNotificationEndpointsListRequest', response_type_name='NotificationEndpointList', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionNotificationEndpointsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNotificationEndpoints.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/notificationEndpoints/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionNotificationEndpointsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)