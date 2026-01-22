from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class OrganizationsAssetsService(base_api.BaseApiService):
    """Service class for the organizations_assets resource."""
    _NAME = 'organizations_assets'

    def __init__(self, client):
        super(SecuritycenterV2.OrganizationsAssetsService, self).__init__(client)
        self._upload_configs = {}

    def UpdateSecurityMarks(self, request, global_params=None):
        """Updates security marks. For Finding Security marks, if no location is specified, finding is assumed to be in global. Assets Security Marks can only be accessed through global endpoint.

      Args:
        request: (SecuritycenterOrganizationsAssetsUpdateSecurityMarksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2SecurityMarks) The response message.
      """
        config = self.GetMethodConfig('UpdateSecurityMarks')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSecurityMarks.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/assets/{assetsId}/securityMarks', http_method='PATCH', method_id='securitycenter.organizations.assets.updateSecurityMarks', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2SecurityMarks', request_type_name='SecuritycenterOrganizationsAssetsUpdateSecurityMarksRequest', response_type_name='GoogleCloudSecuritycenterV2SecurityMarks', supports_download=False)