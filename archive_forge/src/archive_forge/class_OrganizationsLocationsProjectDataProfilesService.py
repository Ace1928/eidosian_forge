from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class OrganizationsLocationsProjectDataProfilesService(base_api.BaseApiService):
    """Service class for the organizations_locations_projectDataProfiles resource."""
    _NAME = 'organizations_locations_projectDataProfiles'

    def __init__(self, client):
        super(DlpV2.OrganizationsLocationsProjectDataProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a project data profile.

      Args:
        request: (DlpOrganizationsLocationsProjectDataProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ProjectDataProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/projectDataProfiles/{projectDataProfilesId}', http_method='GET', method_id='dlp.organizations.locations.projectDataProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpOrganizationsLocationsProjectDataProfilesGetRequest', response_type_name='GooglePrivacyDlpV2ProjectDataProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists project data profiles for an organization.

      Args:
        request: (DlpOrganizationsLocationsProjectDataProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListProjectDataProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/locations/{locationsId}/projectDataProfiles', http_method='GET', method_id='dlp.organizations.locations.projectDataProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/projectDataProfiles', request_field='', request_type_name='DlpOrganizationsLocationsProjectDataProfilesListRequest', response_type_name='GooglePrivacyDlpV2ListProjectDataProfilesResponse', supports_download=False)