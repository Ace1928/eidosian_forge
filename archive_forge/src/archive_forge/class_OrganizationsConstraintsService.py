from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages as messages
class OrganizationsConstraintsService(base_api.BaseApiService):
    """Service class for the organizations_constraints resource."""
    _NAME = 'organizations_constraints'

    def __init__(self, client):
        super(OrgpolicyV2.OrganizationsConstraintsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists constraints that could be applied on the specified resource.

      Args:
        request: (OrgpolicyOrganizationsConstraintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudOrgpolicyV2ListConstraintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/organizations/{organizationsId}/constraints', http_method='GET', method_id='orgpolicy.organizations.constraints.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/constraints', request_field='', request_type_name='OrgpolicyOrganizationsConstraintsListRequest', response_type_name='GoogleCloudOrgpolicyV2ListConstraintsResponse', supports_download=False)