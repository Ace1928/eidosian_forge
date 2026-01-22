from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ScansService(base_api.BaseApiService):
    """Service class for the scans resource."""
    _NAME = 'scans'

    def __init__(self, client):
        super(SpannerV1.ScansService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Return available scans given a Database-specific resource name.

      Args:
        request: (SpannerScansListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScansResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/scans', http_method='GET', method_id='spanner.scans.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}', request_field='', request_type_name='SpannerScansListRequest', response_type_name='ListScansResponse', supports_download=False)