from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NetworksService(base_api.BaseApiService):
    """Service class for the networks resource."""
    _NAME = 'networks'

    def __init__(self, client):
        super(ComputeBeta.NetworksService, self).__init__(client)
        self._upload_configs = {}

    def AddPeering(self, request, global_params=None):
        """Adds a peering to the specified network.

      Args:
        request: (ComputeNetworksAddPeeringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddPeering')
        return self._RunMethod(config, request, global_params=global_params)
    AddPeering.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networks.addPeering', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networks/{network}/addPeering', request_field='networksAddPeeringRequest', request_type_name='ComputeNetworksAddPeeringRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified network.

      Args:
        request: (ComputeNetworksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.networks.delete', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networks/{network}', request_field='', request_type_name='ComputeNetworksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified network.

      Args:
        request: (ComputeNetworksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Network) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networks.get', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=[], relative_path='projects/{project}/global/networks/{network}', request_field='', request_type_name='ComputeNetworksGetRequest', response_type_name='Network', supports_download=False)

    def GetEffectiveFirewalls(self, request, global_params=None):
        """Returns the effective firewalls on a given network.

      Args:
        request: (ComputeNetworksGetEffectiveFirewallsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworksGetEffectiveFirewallsResponse) The response message.
      """
        config = self.GetMethodConfig('GetEffectiveFirewalls')
        return self._RunMethod(config, request, global_params=global_params)
    GetEffectiveFirewalls.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networks.getEffectiveFirewalls', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=[], relative_path='projects/{project}/global/networks/{network}/getEffectiveFirewalls', request_field='', request_type_name='ComputeNetworksGetEffectiveFirewallsRequest', response_type_name='NetworksGetEffectiveFirewallsResponse', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a network in the specified project using the data included in the request.

      Args:
        request: (ComputeNetworksInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networks.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/networks', request_field='network', request_type_name='ComputeNetworksInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of networks available to the specified project.

      Args:
        request: (ComputeNetworksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networks.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/networks', request_field='', request_type_name='ComputeNetworksListRequest', response_type_name='NetworkList', supports_download=False)

    def ListPeeringRoutes(self, request, global_params=None):
        """Lists the peering routes exchanged over peering connection.

      Args:
        request: (ComputeNetworksListPeeringRoutesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExchangedPeeringRoutesList) The response message.
      """
        config = self.GetMethodConfig('ListPeeringRoutes')
        return self._RunMethod(config, request, global_params=global_params)
    ListPeeringRoutes.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networks.listPeeringRoutes', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['direction', 'filter', 'maxResults', 'orderBy', 'pageToken', 'peeringName', 'region', 'returnPartialSuccess'], relative_path='projects/{project}/global/networks/{network}/listPeeringRoutes', request_field='', request_type_name='ComputeNetworksListPeeringRoutesRequest', response_type_name='ExchangedPeeringRoutesList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified network with the data included in the request. Only the following fields can be modified: routingConfig.routingMode.

      Args:
        request: (ComputeNetworksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.networks.patch', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networks/{network}', request_field='networkResource', request_type_name='ComputeNetworksPatchRequest', response_type_name='Operation', supports_download=False)

    def RemovePeering(self, request, global_params=None):
        """Removes a peering from the specified network.

      Args:
        request: (ComputeNetworksRemovePeeringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemovePeering')
        return self._RunMethod(config, request, global_params=global_params)
    RemovePeering.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networks.removePeering', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networks/{network}/removePeering', request_field='networksRemovePeeringRequest', request_type_name='ComputeNetworksRemovePeeringRequest', response_type_name='Operation', supports_download=False)

    def SwitchToCustomMode(self, request, global_params=None):
        """Switches the network mode from auto subnet mode to custom subnet mode.

      Args:
        request: (ComputeNetworksSwitchToCustomModeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SwitchToCustomMode')
        return self._RunMethod(config, request, global_params=global_params)
    SwitchToCustomMode.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networks.switchToCustomMode', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networks/{network}/switchToCustomMode', request_field='', request_type_name='ComputeNetworksSwitchToCustomModeRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeNetworksTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networks.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/networks/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeNetworksTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)

    def UpdatePeering(self, request, global_params=None):
        """Updates the specified network peering with the data included in the request. You can only modify the NetworkPeering.export_custom_routes field and the NetworkPeering.import_custom_routes field.

      Args:
        request: (ComputeNetworksUpdatePeeringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdatePeering')
        return self._RunMethod(config, request, global_params=global_params)
    UpdatePeering.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.networks.updatePeering', ordered_params=['project', 'network'], path_params=['network', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/networks/{network}/updatePeering', request_field='networksUpdatePeeringRequest', request_type_name='ComputeNetworksUpdatePeeringRequest', response_type_name='Operation', supports_download=False)