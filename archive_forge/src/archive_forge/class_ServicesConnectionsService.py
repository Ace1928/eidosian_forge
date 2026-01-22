from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1beta import servicenetworking_v1beta_messages as messages
class ServicesConnectionsService(base_api.BaseApiService):
    """Service class for the services_connections resource."""
    _NAME = 'services_connections'

    def __init__(self, client):
        super(ServicenetworkingV1beta.ServicesConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """To connect service to a VPC network peering connection.
must be established prior to service provisioning.
This method must be invoked by the consumer VPC network administrator
It will establish a permanent peering connection with a shared
network created in the service producer organization and register a
reserved IP range(s) to be used for service subnetwork provisioning.
This connection will be used for all supported services in the service
producer organization, so it only needs to be invoked once.
Operation<response: Connection>.

      Args:
        request: (ServicenetworkingServicesConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/services/{servicesId}/connections', http_method='POST', method_id='servicenetworking.services.connections.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta/{+parent}/connections', request_field='connection', request_type_name='ServicenetworkingServicesConnectionsCreateRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Service consumer use this method to list configured peering connection for.
the given service and consumer network.

      Args:
        request: (ServicenetworkingServicesConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/services/{servicesId}/connections', http_method='GET', method_id='servicenetworking.services.connections.list', ordered_params=['parent'], path_params=['parent'], query_params=['network'], relative_path='v1beta/{+parent}/connections', request_field='', request_type_name='ServicenetworkingServicesConnectionsListRequest', response_type_name='ListConnectionsResponse', supports_download=False)