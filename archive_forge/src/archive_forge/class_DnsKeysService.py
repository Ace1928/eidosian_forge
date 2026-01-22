from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
class DnsKeysService(base_api.BaseApiService):
    """Service class for the dnsKeys resource."""
    _NAME = 'dnsKeys'

    def __init__(self, client):
        super(DnsV1alpha2.DnsKeysService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Fetches the representation of an existing DnsKey.

      Args:
        request: (DnsDnsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.dnsKeys.get', ordered_params=['project', 'managedZone', 'dnsKeyId'], path_params=['dnsKeyId', 'managedZone', 'project'], query_params=['clientOperationId', 'digestType'], relative_path='dns/v1alpha2/projects/{project}/managedZones/{managedZone}/dnsKeys/{dnsKeyId}', request_field='', request_type_name='DnsDnsKeysGetRequest', response_type_name='DnsKey', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerates DnsKeys to a ResourceRecordSet collection.

      Args:
        request: (DnsDnsKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsKeysListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.dnsKeys.list', ordered_params=['project', 'managedZone'], path_params=['managedZone', 'project'], query_params=['digestType', 'maxResults', 'pageToken'], relative_path='dns/v1alpha2/projects/{project}/managedZones/{managedZone}/dnsKeys', request_field='', request_type_name='DnsDnsKeysListRequest', response_type_name='DnsKeysListResponse', supports_download=False)