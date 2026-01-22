from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsSecurityReportsService(base_api.BaseApiService):
    """Service class for the organizations_environments_securityReports resource."""
    _NAME = 'organizations_environments_securityReports'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsSecurityReportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Submit a report request to be processed in the background. If the submission succeeds, the API returns a 200 status and an ID that refer to the report request. In addition to the HTTP status 200, the `state` of "enqueued" means that the request succeeded.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityReportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReport) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityReports', http_method='POST', method_id='apigee.organizations.environments.securityReports.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/securityReports', request_field='googleCloudApigeeV1SecurityReportQuery', request_type_name='ApigeeOrganizationsEnvironmentsSecurityReportsCreateRequest', response_type_name='GoogleCloudApigeeV1SecurityReport', supports_download=False)

    def Get(self, request, global_params=None):
        """Get security report status If the query is still in progress, the `state` is set to "running" After the query has completed successfully, `state` is set to "completed".

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityReports/{securityReportsId}', http_method='GET', method_id='apigee.organizations.environments.securityReports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityReportsGetRequest', response_type_name='GoogleCloudApigeeV1SecurityReport', supports_download=False)

    def GetResult(self, request, global_params=None):
        """After the query is completed, use this API to retrieve the results as file. If the request succeeds, and there is a non-zero result set, the result is downloaded to the client as a zipped JSON file. The name of the downloaded file will be: OfflineQueryResult-.zip Example: `OfflineQueryResult-9cfc0d85-0f30-46d6-ae6f-318d0cb961bd.zip`.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityReportsGetResultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('GetResult')
        return self._RunMethod(config, request, global_params=global_params)
    GetResult.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityReports/{securityReportsId}/result', http_method='GET', method_id='apigee.organizations.environments.securityReports.getResult', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityReportsGetResultRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def GetResultView(self, request, global_params=None):
        """After the query is completed, use this API to view the query result when result size is small.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityReportsGetResultViewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReportResultView) The response message.
      """
        config = self.GetMethodConfig('GetResultView')
        return self._RunMethod(config, request, global_params=global_params)
    GetResultView.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityReports/{securityReportsId}/resultView', http_method='GET', method_id='apigee.organizations.environments.securityReports.getResultView', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityReportsGetResultViewRequest', response_type_name='GoogleCloudApigeeV1SecurityReportResultView', supports_download=False)

    def List(self, request, global_params=None):
        """Return a list of Security Reports.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityReports', http_method='GET', method_id='apigee.organizations.environments.securityReports.list', ordered_params=['parent'], path_params=['parent'], query_params=['dataset', 'from_', 'pageSize', 'pageToken', 'status', 'submittedBy', 'to'], relative_path='v1/{+parent}/securityReports', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityReportsListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityReportsResponse', supports_download=False)