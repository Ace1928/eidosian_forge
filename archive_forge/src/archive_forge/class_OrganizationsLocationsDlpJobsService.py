from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class OrganizationsLocationsDlpJobsService(base_api.BaseApiService):
    """Service class for the organizations_locations_dlpJobs resource."""
    _NAME = 'organizations_locations_dlpJobs'

    def __init__(self, client):
        super(DlpV2.OrganizationsLocationsDlpJobsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists DlpJobs that match the specified filter in the request. See https://cloud.google.com/sensitive-data-protection/docs/inspecting-storage and https://cloud.google.com/sensitive-data-protection/docs/compute-risk-analysis to learn more.

      Args:
        request: (DlpOrganizationsLocationsDlpJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListDlpJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/dlpJobs', http_method='GET', method_id='dlp.organizations.locations.dlpJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type'], relative_path='v2/{+parent}/dlpJobs', request_field='', request_type_name='DlpOrganizationsLocationsDlpJobsListRequest', response_type_name='GooglePrivacyDlpV2ListDlpJobsResponse', supports_download=False)