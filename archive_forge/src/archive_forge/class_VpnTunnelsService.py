from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class VpnTunnelsService(base_api.BaseApiService):
    """Service class for the vpnTunnels resource."""
    _NAME = 'vpnTunnels'

    def __init__(self, client):
        super(ComputeBeta.VpnTunnelsService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of VPN tunnels. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeVpnTunnelsAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnTunnelAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnTunnels.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/vpnTunnels', request_field='', request_type_name='ComputeVpnTunnelsAggregatedListRequest', response_type_name='VpnTunnelAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified VpnTunnel resource.

      Args:
        request: (ComputeVpnTunnelsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.vpnTunnels.delete', ordered_params=['project', 'region', 'vpnTunnel'], path_params=['project', 'region', 'vpnTunnel'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/vpnTunnels/{vpnTunnel}', request_field='', request_type_name='ComputeVpnTunnelsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified VpnTunnel resource.

      Args:
        request: (ComputeVpnTunnelsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnTunnel) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnTunnels.get', ordered_params=['project', 'region', 'vpnTunnel'], path_params=['project', 'region', 'vpnTunnel'], query_params=[], relative_path='projects/{project}/regions/{region}/vpnTunnels/{vpnTunnel}', request_field='', request_type_name='ComputeVpnTunnelsGetRequest', response_type_name='VpnTunnel', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a VpnTunnel resource in the specified project and region using the data included in the request.

      Args:
        request: (ComputeVpnTunnelsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.vpnTunnels.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/vpnTunnels', request_field='vpnTunnel', request_type_name='ComputeVpnTunnelsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of VpnTunnel resources contained in the specified project and region.

      Args:
        request: (ComputeVpnTunnelsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnTunnelList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnTunnels.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/vpnTunnels', request_field='', request_type_name='ComputeVpnTunnelsListRequest', response_type_name='VpnTunnelList', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a VpnTunnel. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeVpnTunnelsSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.vpnTunnels.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/vpnTunnels/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeVpnTunnelsSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeVpnTunnelsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.vpnTunnels.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/vpnTunnels/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeVpnTunnelsTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)