from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsAnalyticsExportsService(base_api.BaseApiService):
    """Service class for the organizations_environments_analytics_exports resource."""
    _NAME = 'organizations_environments_analytics_exports'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsAnalyticsExportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Submit a data export job to be processed in the background. If the request is successful, the API returns a 201 status, a URI that can be used to retrieve the status of the export job, and the `state` value of "enqueued".

      Args:
        request: (ApigeeOrganizationsEnvironmentsAnalyticsExportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Export) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/analytics/exports', http_method='POST', method_id='apigee.organizations.environments.analytics.exports.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/analytics/exports', request_field='googleCloudApigeeV1ExportRequest', request_type_name='ApigeeOrganizationsEnvironmentsAnalyticsExportsCreateRequest', response_type_name='GoogleCloudApigeeV1Export', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details and status of an analytics export job. If the export job is still in progress, its `state` is set to "running". After the export job has completed successfully, its `state` is set to "completed". If the export job fails, its `state` is set to `failed`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsAnalyticsExportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Export) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/analytics/exports/{exportsId}', http_method='GET', method_id='apigee.organizations.environments.analytics.exports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsAnalyticsExportsGetRequest', response_type_name='GoogleCloudApigeeV1Export', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the details and status of all analytics export jobs belonging to the parent organization and environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsAnalyticsExportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListExportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/analytics/exports', http_method='GET', method_id='apigee.organizations.environments.analytics.exports.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/analytics/exports', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsAnalyticsExportsListRequest', response_type_name='GoogleCloudApigeeV1ListExportsResponse', supports_download=False)