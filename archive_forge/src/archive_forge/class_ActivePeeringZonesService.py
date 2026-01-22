from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
class ActivePeeringZonesService(base_api.BaseApiService):
    """Service class for the activePeeringZones resource."""
    _NAME = 'activePeeringZones'

    def __init__(self, client):
        super(DnsV1alpha2.ActivePeeringZonesService, self).__init__(client)
        self._upload_configs = {}

    def Deactivate(self, request, global_params=None):
        """Deactivates a Peering Zone if it's not already deactivated. Returns an error if the managed zone cannot be found, is not a peering zone. If the zone is already deactivated, returns false for deactivate_succeeded field.

      Args:
        request: (DnsActivePeeringZonesDeactivateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PeeringZoneDeactivateResponse) The response message.
      """
        config = self.GetMethodConfig('Deactivate')
        return self._RunMethod(config, request, global_params=global_params)
    Deactivate.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='dns.activePeeringZones.deactivate', ordered_params=['project', 'peeringZoneId'], path_params=['peeringZoneId', 'project'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/activePeeringZones/{peeringZoneId}', request_field='', request_type_name='DnsActivePeeringZonesDeactivateRequest', response_type_name='PeeringZoneDeactivateResponse', supports_download=False)

    def GetPeeringZoneInfo(self, request, global_params=None):
        """Fetches the representation of an existing PeeringZone.

      Args:
        request: (DnsActivePeeringZonesGetPeeringZoneInfoRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
        config = self.GetMethodConfig('GetPeeringZoneInfo')
        return self._RunMethod(config, request, global_params=global_params)
    GetPeeringZoneInfo.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.activePeeringZones.getPeeringZoneInfo', ordered_params=['project', 'peeringZoneId'], path_params=['peeringZoneId', 'project'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/activePeeringZones/{peeringZoneId}', request_field='', request_type_name='DnsActivePeeringZonesGetPeeringZoneInfoRequest', response_type_name='ManagedZone', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerates PeeringZones that target a given network through DNS peering.

      Args:
        request: (DnsActivePeeringZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PeeringZonesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.activePeeringZones.list', ordered_params=['project', 'targetNetwork'], path_params=['project'], query_params=['maxResults', 'pageToken', 'targetNetwork'], relative_path='dns/v1alpha2/projects/{project}/activePeeringZones', request_field='', request_type_name='DnsActivePeeringZonesListRequest', response_type_name='PeeringZonesListResponse', supports_download=False)