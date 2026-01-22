from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class OrganizationsLocationsRecentQueriesService(base_api.BaseApiService):
    """Service class for the organizations_locations_recentQueries resource."""
    _NAME = 'organizations_locations_recentQueries'

    def __init__(self, client):
        super(LoggingV2.OrganizationsLocationsRecentQueriesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the RecentQueries that were created by the user making the request.

      Args:
        request: (LoggingOrganizationsLocationsRecentQueriesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRecentQueriesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/recentQueries', http_method='GET', method_id='logging.organizations.locations.recentQueries.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/recentQueries', request_field='', request_type_name='LoggingOrganizationsLocationsRecentQueriesListRequest', response_type_name='ListRecentQueriesResponse', supports_download=False)