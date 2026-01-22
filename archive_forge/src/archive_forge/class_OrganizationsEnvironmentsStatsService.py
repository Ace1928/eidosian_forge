from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsStatsService(base_api.BaseApiService):
    """Service class for the organizations_environments_stats resource."""
    _NAME = 'organizations_environments_stats'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsStatsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieve metrics grouped by dimensions. The types of metrics you can retrieve include traffic, message counts, API call latency, response size, and cache hits and counts. Dimensions let you view metrics in meaningful groups. You can optionally pass dimensions as path parameters to the `stats` API. If dimensions are not specified, the metrics are computed on the entire set of data for the given time range.

      Args:
        request: (ApigeeOrganizationsEnvironmentsStatsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Stats) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/stats/{statsId}', http_method='GET', method_id='apigee.organizations.environments.stats.get', ordered_params=['name'], path_params=['name'], query_params=['accuracy', 'aggTable', 'filter', 'limit', 'offset', 'realtime', 'select', 'sonar', 'sort', 'sortby', 'timeRange', 'timeUnit', 'topk', 'tsAscending', 'tzo'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsStatsGetRequest', response_type_name='GoogleCloudApigeeV1Stats', supports_download=False)