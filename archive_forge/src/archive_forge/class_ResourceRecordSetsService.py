from __future__ import absolute_import
from apitools.base.py import base_api
from samples.dns_sample.dns_v1 import dns_v1_messages as messages
class ResourceRecordSetsService(base_api.BaseApiService):
    """Service class for the resourceRecordSets resource."""
    _NAME = u'resourceRecordSets'

    def __init__(self, client):
        super(DnsV1.ResourceRecordSetsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Enumerate ResourceRecordSets that have been created but not yet deleted.

      Args:
        request: (DnsResourceRecordSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceRecordSetsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'dns.resourceRecordSets.list', ordered_params=[u'project', u'managedZone'], path_params=[u'managedZone', u'project'], query_params=[u'maxResults', u'name', u'pageToken', u'type'], relative_path=u'projects/{project}/managedZones/{managedZone}/rrsets', request_field='', request_type_name=u'DnsResourceRecordSetsListRequest', response_type_name=u'ResourceRecordSetsListResponse', supports_download=False)