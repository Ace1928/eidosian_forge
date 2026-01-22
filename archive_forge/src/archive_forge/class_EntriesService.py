from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datacatalog.v1 import datacatalog_v1_messages as messages
class EntriesService(base_api.BaseApiService):
    """Service class for the entries resource."""
    _NAME = 'entries'

    def __init__(self, client):
        super(DatacatalogV1.EntriesService, self).__init__(client)
        self._upload_configs = {}

    def Lookup(self, request, global_params=None):
        """Gets an entry by its target resource name. The resource name comes from the source Google Cloud Platform service.

      Args:
        request: (DatacatalogEntriesLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDatacatalogV1Entry) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='datacatalog.entries.lookup', ordered_params=[], path_params=[], query_params=['fullyQualifiedName', 'linkedResource', 'location', 'project', 'sqlResource'], relative_path='v1/entries:lookup', request_field='', request_type_name='DatacatalogEntriesLookupRequest', response_type_name='GoogleCloudDatacatalogV1Entry', supports_download=False)