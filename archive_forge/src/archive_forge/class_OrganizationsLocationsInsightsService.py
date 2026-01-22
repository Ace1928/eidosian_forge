from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class OrganizationsLocationsInsightsService(base_api.BaseApiService):
    """Service class for the organizations_locations_insights resource."""
    _NAME = 'organizations_locations_insights'

    def __init__(self, client):
        super(BeyondcorpV1alpha.OrganizationsLocationsInsightsService, self).__init__(client)
        self._upload_configs = {}

    def ConfiguredInsight(self, request, global_params=None):
        """Gets the value for a selected particular insight based on the provided filters. Use the organization level path for fetching at org level and project level path for fetching the insight value specific to a particular project.

      Args:
        request: (BeyondcorpOrganizationsLocationsInsightsConfiguredInsightRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformInsightsV1alphaConfiguredInsightResponse) The response message.
      """
        config = self.GetMethodConfig('ConfiguredInsight')
        return self._RunMethod(config, request, global_params=global_params)
    ConfiguredInsight.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/insights/{insightsId}:configuredInsight', http_method='GET', method_id='beyondcorp.organizations.locations.insights.configuredInsight', ordered_params=['insight'], path_params=['insight'], query_params=['aggregation', 'customGrouping_fieldFilter', 'customGrouping_groupFields', 'endTime', 'fieldFilter', 'group', 'pageSize', 'pageToken', 'startTime'], relative_path='v1alpha/{+insight}:configuredInsight', request_field='', request_type_name='BeyondcorpOrganizationsLocationsInsightsConfiguredInsightRequest', response_type_name='GoogleCloudBeyondcorpSaasplatformInsightsV1alphaConfiguredInsightResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the value for a selected particular insight with default configuration. The default aggregation level is 'DAILY' and no grouping will be applied or default grouping if applicable. The data will be returned for recent 7 days starting the day before. The insight data size will be limited to 50 rows. Use the organization level path for fetching at org level and project level path for fetching the insight value specific to a particular project. Setting the `view` to `BASIC` will only return the metadata for the insight.

      Args:
        request: (BeyondcorpOrganizationsLocationsInsightsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsight) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/insights/{insightsId}', http_method='GET', method_id='beyondcorp.organizations.locations.insights.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpOrganizationsLocationsInsightsGetRequest', response_type_name='GoogleCloudBeyondcorpSaasplatformInsightsV1alphaInsight', supports_download=False)

    def List(self, request, global_params=None):
        """Lists for all the available insights that could be fetched from the system. Allows to filter using category. Setting the `view` to `BASIC` will let you iterate over the list of insight metadatas.

      Args:
        request: (BeyondcorpOrganizationsLocationsInsightsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpSaasplatformInsightsV1alphaListInsightsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/{locationsId}/insights', http_method='GET', method_id='beyondcorp.organizations.locations.insights.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1alpha/{+parent}/insights', request_field='', request_type_name='BeyondcorpOrganizationsLocationsInsightsListRequest', response_type_name='GoogleCloudBeyondcorpSaasplatformInsightsV1alphaListInsightsResponse', supports_download=False)