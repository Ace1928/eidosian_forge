from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesProjectsGlobalNetworksDnsZonesService(base_api.BaseApiService):
    """Service class for the services_projects_global_networks_dnsZones resource."""
    _NAME = 'services_projects_global_networks_dnsZones'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesProjectsGlobalNetworksDnsZonesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Service producers can use this method to retrieve a DNS zone in the shared producer host project and the matching peering zones in consumer project.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksDnsZonesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetDnsZoneResponse) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}/dnsZones/{dnsZonesId}', http_method='GET', method_id='servicenetworking.services.projects.global.networks.dnsZones.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksDnsZonesGetRequest', response_type_name='GetDnsZoneResponse', supports_download=False)

    def List(self, request, global_params=None):
        """* Service producers can use this method to retrieve a list of available DNS zones in the shared producer host project and the matching peering zones in the consumer project. *.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksDnsZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDnsZonesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}/dnsZones:list', http_method='GET', method_id='servicenetworking.services.projects.global.networks.dnsZones.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dnsZones:list', request_field='', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksDnsZonesListRequest', response_type_name='ListDnsZonesResponse', supports_download=False)