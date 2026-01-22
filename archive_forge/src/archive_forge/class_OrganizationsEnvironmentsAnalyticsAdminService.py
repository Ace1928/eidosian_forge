from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsAnalyticsAdminService(base_api.BaseApiService):
    """Service class for the organizations_environments_analytics_admin resource."""
    _NAME = 'organizations_environments_analytics_admin'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsAnalyticsAdminService, self).__init__(client)
        self._upload_configs = {}

    def GetSchemav2(self, request, global_params=None):
        """Gets a list of metrics and dimensions that can be used to create analytics queries and reports. Each schema element contains the name of the field, its associated type, and a flag indicating whether it is a standard or custom field.

      Args:
        request: (ApigeeOrganizationsEnvironmentsAnalyticsAdminGetSchemav2Request) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Schema) The response message.
      """
        config = self.GetMethodConfig('GetSchemav2')
        return self._RunMethod(config, request, global_params=global_params)
    GetSchemav2.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/analytics/admin/schemav2', http_method='GET', method_id='apigee.organizations.environments.analytics.admin.getSchemav2', ordered_params=['name'], path_params=['name'], query_params=['disableCache', 'type'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsAnalyticsAdminGetSchemav2Request', response_type_name='GoogleCloudApigeeV1Schema', supports_download=False)