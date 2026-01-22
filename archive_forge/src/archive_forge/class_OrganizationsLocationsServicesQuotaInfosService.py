from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudquotas.v1 import cloudquotas_v1_messages as messages
class OrganizationsLocationsServicesQuotaInfosService(base_api.BaseApiService):
    """Service class for the organizations_locations_services_quotaInfos resource."""
    _NAME = 'organizations_locations_services_quotaInfos'

    def __init__(self, client):
        super(CloudquotasV1.OrganizationsLocationsServicesQuotaInfosService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieve the QuotaInfo of a quota for a project, folder or organization.

      Args:
        request: (CloudquotasOrganizationsLocationsServicesQuotaInfosGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QuotaInfo) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/services/{servicesId}/quotaInfos/{quotaInfosId}', http_method='GET', method_id='cloudquotas.organizations.locations.services.quotaInfos.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudquotasOrganizationsLocationsServicesQuotaInfosGetRequest', response_type_name='QuotaInfo', supports_download=False)

    def List(self, request, global_params=None):
        """Lists QuotaInfos of all quotas for a given project, folder or organization.

      Args:
        request: (CloudquotasOrganizationsLocationsServicesQuotaInfosListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListQuotaInfosResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/locations/{locationsId}/services/{servicesId}/quotaInfos', http_method='GET', method_id='cloudquotas.organizations.locations.services.quotaInfos.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/quotaInfos', request_field='', request_type_name='CloudquotasOrganizationsLocationsServicesQuotaInfosListRequest', response_type_name='ListQuotaInfosResponse', supports_download=False)