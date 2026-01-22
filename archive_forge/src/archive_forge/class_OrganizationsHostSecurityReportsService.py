from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsHostSecurityReportsService(base_api.BaseApiService):
    """Service class for the organizations_hostSecurityReports resource."""
    _NAME = 'organizations_hostSecurityReports'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsHostSecurityReportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Submit a query at host level to be processed in the background. If the submission of the query succeeds, the API returns a 201 status and an ID that refer to the query. In addition to the HTTP status 201, the `state` of "enqueued" means that the request succeeded.

      Args:
        request: (ApigeeOrganizationsHostSecurityReportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReport) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostSecurityReports', http_method='POST', method_id='apigee.organizations.hostSecurityReports.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/hostSecurityReports', request_field='googleCloudApigeeV1SecurityReportQuery', request_type_name='ApigeeOrganizationsHostSecurityReportsCreateRequest', response_type_name='GoogleCloudApigeeV1SecurityReport', supports_download=False)

    def Get(self, request, global_params=None):
        """Get status of a query submitted at host level. If the query is still in progress, the `state` is set to "running" After the query has completed successfully, `state` is set to "completed".

      Args:
        request: (ApigeeOrganizationsHostSecurityReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostSecurityReports/{hostSecurityReportsId}', http_method='GET', method_id='apigee.organizations.hostSecurityReports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsHostSecurityReportsGetRequest', response_type_name='GoogleCloudApigeeV1SecurityReport', supports_download=False)

    def GetResult(self, request, global_params=None):
        """After the query is completed, use this API to retrieve the results. If the request succeeds, and there is a non-zero result set, the result is downloaded to the client as a zipped JSON file. The name of the downloaded file will be: OfflineQueryResult-.zip Example: `OfflineQueryResult-9cfc0d85-0f30-46d6-ae6f-318d0cb961bd.zip`.

      Args:
        request: (ApigeeOrganizationsHostSecurityReportsGetResultRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('GetResult')
        return self._RunMethod(config, request, global_params=global_params)
    GetResult.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostSecurityReports/{hostSecurityReportsId}/result', http_method='GET', method_id='apigee.organizations.hostSecurityReports.getResult', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsHostSecurityReportsGetResultRequest', response_type_name='GoogleApiHttpBody', supports_download=False)

    def GetResultView(self, request, global_params=None):
        """After the query is completed, use this API to view the query result when result size is small.

      Args:
        request: (ApigeeOrganizationsHostSecurityReportsGetResultViewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReportResultView) The response message.
      """
        config = self.GetMethodConfig('GetResultView')
        return self._RunMethod(config, request, global_params=global_params)
    GetResultView.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostSecurityReports/{hostSecurityReportsId}/resultView', http_method='GET', method_id='apigee.organizations.hostSecurityReports.getResultView', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsHostSecurityReportsGetResultViewRequest', response_type_name='GoogleCloudApigeeV1SecurityReportResultView', supports_download=False)

    def List(self, request, global_params=None):
        """Return a list of Security Reports at host level.

      Args:
        request: (ApigeeOrganizationsHostSecurityReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/hostSecurityReports', http_method='GET', method_id='apigee.organizations.hostSecurityReports.list', ordered_params=['parent'], path_params=['parent'], query_params=['dataset', 'envgroupHostname', 'from_', 'pageSize', 'pageToken', 'status', 'submittedBy', 'to'], relative_path='v1/{+parent}/hostSecurityReports', request_field='', request_type_name='ApigeeOrganizationsHostSecurityReportsListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityReportsResponse', supports_download=False)