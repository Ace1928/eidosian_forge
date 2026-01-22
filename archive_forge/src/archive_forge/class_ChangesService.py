from __future__ import absolute_import
from apitools.base.py import base_api
from samples.dns_sample.dns_v1 import dns_v1_messages as messages
class ChangesService(base_api.BaseApiService):
    """Service class for the changes resource."""
    _NAME = u'changes'

    def __init__(self, client):
        super(DnsV1.ChangesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Atomically update the ResourceRecordSet collection.

      Args:
        request: (DnsChangesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Change) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'dns.changes.create', ordered_params=[u'project', u'managedZone'], path_params=[u'managedZone', u'project'], query_params=[], relative_path=u'projects/{project}/managedZones/{managedZone}/changes', request_field=u'change', request_type_name=u'DnsChangesCreateRequest', response_type_name=u'Change', supports_download=False)

    def Get(self, request, global_params=None):
        """Fetch the representation of an existing Change.

      Args:
        request: (DnsChangesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Change) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'dns.changes.get', ordered_params=[u'project', u'managedZone', u'changeId'], path_params=[u'changeId', u'managedZone', u'project'], query_params=[], relative_path=u'projects/{project}/managedZones/{managedZone}/changes/{changeId}', request_field='', request_type_name=u'DnsChangesGetRequest', response_type_name=u'Change', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerate Changes to a ResourceRecordSet collection.

      Args:
        request: (DnsChangesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ChangesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'dns.changes.list', ordered_params=[u'project', u'managedZone'], path_params=[u'managedZone', u'project'], query_params=[u'maxResults', u'pageToken', u'sortBy', u'sortOrder'], relative_path=u'projects/{project}/managedZones/{managedZone}/changes', request_field='', request_type_name=u'DnsChangesListRequest', response_type_name=u'ChangesListResponse', supports_download=False)