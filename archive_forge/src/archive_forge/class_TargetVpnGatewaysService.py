from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class TargetVpnGatewaysService(base_api.BaseApiService):
    """Service class for the targetVpnGateways resource."""
    _NAME = 'targetVpnGateways'

    def __init__(self, client):
        super(ComputeBeta.TargetVpnGatewaysService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of target VPN gateways. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeTargetVpnGatewaysAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetVpnGatewayAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetVpnGateways.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/targetVpnGateways', request_field='', request_type_name='ComputeTargetVpnGatewaysAggregatedListRequest', response_type_name='TargetVpnGatewayAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified target VPN gateway.

      Args:
        request: (ComputeTargetVpnGatewaysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.targetVpnGateways.delete', ordered_params=['project', 'region', 'targetVpnGateway'], path_params=['project', 'region', 'targetVpnGateway'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetVpnGateways/{targetVpnGateway}', request_field='', request_type_name='ComputeTargetVpnGatewaysDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified target VPN gateway.

      Args:
        request: (ComputeTargetVpnGatewaysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetVpnGateway) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetVpnGateways.get', ordered_params=['project', 'region', 'targetVpnGateway'], path_params=['project', 'region', 'targetVpnGateway'], query_params=[], relative_path='projects/{project}/regions/{region}/targetVpnGateways/{targetVpnGateway}', request_field='', request_type_name='ComputeTargetVpnGatewaysGetRequest', response_type_name='TargetVpnGateway', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a target VPN gateway in the specified project and region using the data included in the request.

      Args:
        request: (ComputeTargetVpnGatewaysInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetVpnGateways.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetVpnGateways', request_field='targetVpnGateway', request_type_name='ComputeTargetVpnGatewaysInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of target VPN gateways available to the specified project and region.

      Args:
        request: (ComputeTargetVpnGatewaysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TargetVpnGatewayList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.targetVpnGateways.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/targetVpnGateways', request_field='', request_type_name='ComputeTargetVpnGatewaysListRequest', response_type_name='TargetVpnGatewayList', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a TargetVpnGateway. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeTargetVpnGatewaysSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetVpnGateways.setLabels', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/targetVpnGateways/{resource}/setLabels', request_field='regionSetLabelsRequest', request_type_name='ComputeTargetVpnGatewaysSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeTargetVpnGatewaysTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.targetVpnGateways.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/targetVpnGateways/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeTargetVpnGatewaysTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)