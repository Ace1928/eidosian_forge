from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class VpnGatewaysService(base_api.BaseApiService):
    """Service class for the vpnGateways resource."""
    _NAME = 'vpnGateways'

    def __init__(self, client):
        super(ComputeBeta.VpnGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of VPN gateways. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeVpnGatewaysAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnGatewayAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnGateways.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/vpnGateways', request_field='', request_type_name='ComputeVpnGatewaysAggregatedListRequest', response_type_name='VpnGatewayAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified VPN gateway.

      Args:
        request: (ComputeVpnGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.vpnGateways.delete', ordered_params=['project', 'region', 'vpnGateway'], path_params=['project', 'region', 'vpnGateway'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/vpnGateways/{vpnGateway}', request_field='', request_type_name='ComputeVpnGatewaysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified VPN gateway.

      Args:
        request: (ComputeVpnGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnGateways.get', ordered_params=['project', 'region', 'vpnGateway'], path_params=['project', 'region', 'vpnGateway'], query_params=[], relative_path='projects/{project}/regions/{region}/vpnGateways/{vpnGateway}', request_field='', request_type_name='ComputeVpnGatewaysGetRequest', response_type_name='VpnGateway', supports_download=False)

    def GetStatus(self, request, global_params=None):
        """Returns the status for the specified VPN gateway.

      Args:
        request: (ComputeVpnGatewaysGetStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnGatewaysGetStatusResponse) The response message.
      """
        config = self.GetMethodConfig('GetStatus')
        return self._RunMethod(config, request, global_params=global_params)
    GetStatus.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnGateways.getStatus', ordered_params=['project', 'region', 'vpnGateway'], path_params=['project', 'region', 'vpnGateway'], query_params=[], relative_path='projects/{project}/regions/{region}/vpnGateways/{vpnGateway}/getStatus', request_field='', request_type_name='ComputeVpnGatewaysGetStatusRequest', response_type_name='VpnGatewaysGetStatusResponse', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a VPN gateway in the specified project and region using the data included in the request.

      Args:
        request: (ComputeVpnGatewaysInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.vpnGateways.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/vpnGateways', request_field='vpnGateway', request_type_name='ComputeVpnGatewaysInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of VPN gateways available to the specified project and region.

      Args:
        request: (ComputeVpnGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpnGatewayList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.vpnGateways.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/vpnGateways', request_field='', request_type_name='ComputeVpnGatewaysListRequest', response_type_name='VpnGatewayList', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a VpnGateway. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeVpnGatewaysSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.vpnGateways.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/vpnGateways/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeVpnGatewaysSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeVpnGatewaysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.vpnGateways.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/vpnGateways/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeVpnGatewaysTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)