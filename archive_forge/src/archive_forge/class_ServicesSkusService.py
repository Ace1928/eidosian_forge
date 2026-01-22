from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbilling.v1 import cloudbilling_v1_messages as messages
class ServicesSkusService(base_api.BaseApiService):
    """Service class for the services_skus resource."""
    _NAME = 'services_skus'

    def __init__(self, client):
        super(CloudbillingV1.ServicesSkusService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists all publicly available SKUs for a given cloud service.

      Args:
        request: (CloudbillingServicesSkusListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSkusResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/skus', http_method='GET', method_id='cloudbilling.services.skus.list', ordered_params=['parent'], path_params=['parent'], query_params=['currencyCode', 'endTime', 'pageSize', 'pageToken', 'startTime'], relative_path='v1/{+parent}/skus', request_field='', request_type_name='CloudbillingServicesSkusListRequest', response_type_name='ListSkusResponse', supports_download=False)