from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recommender.v1alpha2 import recommender_v1alpha2_messages as messages
class BillingAccountsLocationsService(base_api.BaseApiService):
    """Service class for the billingAccounts_locations resource."""
    _NAME = 'billingAccounts_locations'

    def __init__(self, client):
        super(RecommenderV1alpha2.BillingAccountsLocationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists locations with recommendations or insights.

      Args:
        request: (RecommenderBillingAccountsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudLocationListLocationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/billingAccounts/{billingAccountsId}/locations', http_method='GET', method_id='recommender.billingAccounts.locations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+name}/locations', request_field='', request_type_name='RecommenderBillingAccountsLocationsListRequest', response_type_name='GoogleCloudLocationListLocationsResponse', supports_download=False)