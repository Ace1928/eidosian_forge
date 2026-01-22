from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesDnsRecordSetsService(base_api.BaseApiService):
    """Service class for the services_dnsRecordSets resource."""
    _NAME = 'services_dnsRecordSets'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesDnsRecordSetsService, self).__init__(client)
        self._upload_configs = {}

    def Add(self, request, global_params=None):
        """Service producers can use this method to add DNS record sets to private DNS zones in the shared producer host project.

      Args:
        request: (ServicenetworkingServicesDnsRecordSetsAddRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Add')
        return self._RunMethod(config, request, global_params=global_params)
    Add.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsRecordSets:add', http_method='POST', method_id='servicenetworking.services.dnsRecordSets.add', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dnsRecordSets:add', request_field='addDnsRecordSetRequest', request_type_name='ServicenetworkingServicesDnsRecordSetsAddRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Producers can use this method to retrieve information about the DNS record set added to the private zone inside the shared tenant host project associated with a consumer network.

      Args:
        request: (ServicenetworkingServicesDnsRecordSetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsRecordSet) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsRecordSets:get', http_method='GET', method_id='servicenetworking.services.dnsRecordSets.get', ordered_params=['parent'], path_params=['parent'], query_params=['consumerNetwork', 'domain', 'type', 'zone'], relative_path='v1/{+parent}/dnsRecordSets:get', request_field='', request_type_name='ServicenetworkingServicesDnsRecordSetsGetRequest', response_type_name='DnsRecordSet', supports_download=False)

    def List(self, request, global_params=None):
        """Producers can use this method to retrieve a list of available DNS RecordSets available inside the private zone on the tenant host project accessible from their network.

      Args:
        request: (ServicenetworkingServicesDnsRecordSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDnsRecordSetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsRecordSets:list', http_method='GET', method_id='servicenetworking.services.dnsRecordSets.list', ordered_params=['parent'], path_params=['parent'], query_params=['consumerNetwork', 'zone'], relative_path='v1/{+parent}/dnsRecordSets:list', request_field='', request_type_name='ServicenetworkingServicesDnsRecordSetsListRequest', response_type_name='ListDnsRecordSetsResponse', supports_download=False)

    def Remove(self, request, global_params=None):
        """Service producers can use this method to remove DNS record sets from private DNS zones in the shared producer host project.

      Args:
        request: (ServicenetworkingServicesDnsRecordSetsRemoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Remove')
        return self._RunMethod(config, request, global_params=global_params)
    Remove.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsRecordSets:remove', http_method='POST', method_id='servicenetworking.services.dnsRecordSets.remove', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dnsRecordSets:remove', request_field='removeDnsRecordSetRequest', request_type_name='ServicenetworkingServicesDnsRecordSetsRemoveRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Service producers can use this method to update DNS record sets from private DNS zones in the shared producer host project.

      Args:
        request: (ServicenetworkingServicesDnsRecordSetsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsRecordSets:update', http_method='POST', method_id='servicenetworking.services.dnsRecordSets.update', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dnsRecordSets:update', request_field='updateDnsRecordSetRequest', request_type_name='ServicenetworkingServicesDnsRecordSetsUpdateRequest', response_type_name='Operation', supports_download=False)