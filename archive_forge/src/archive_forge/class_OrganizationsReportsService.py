from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsReportsService(base_api.BaseApiService):
    """Service class for the organizations_reports resource."""
    _NAME = 'organizations_reports'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsReportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Custom Report for an Organization. A Custom Report provides Apigee Customers to create custom dashboards in addition to the standard dashboards which are provided. The Custom Report in its simplest form contains specifications about metrics, dimensions and filters. It is important to note that the custom report by itself does not provide an executable entity. The Edge UI converts the custom report definition into an analytics query and displays the result in a chart.

      Args:
        request: (ApigeeOrganizationsReportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1CustomReport) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/reports', http_method='POST', method_id='apigee.organizations.reports.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/reports', request_field='googleCloudApigeeV1CustomReport', request_type_name='ApigeeOrganizationsReportsCreateRequest', response_type_name='GoogleCloudApigeeV1CustomReport', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing custom report definition.

      Args:
        request: (ApigeeOrganizationsReportsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeleteCustomReportResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/reports/{reportsId}', http_method='DELETE', method_id='apigee.organizations.reports.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsReportsDeleteRequest', response_type_name='GoogleCloudApigeeV1DeleteCustomReportResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve a custom report definition.

      Args:
        request: (ApigeeOrganizationsReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1CustomReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/reports/{reportsId}', http_method='GET', method_id='apigee.organizations.reports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsReportsGetRequest', response_type_name='GoogleCloudApigeeV1CustomReport', supports_download=False)

    def List(self, request, global_params=None):
        """Return a list of Custom Reports.

      Args:
        request: (ApigeeOrganizationsReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListCustomReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/reports', http_method='GET', method_id='apigee.organizations.reports.list', ordered_params=['parent'], path_params=['parent'], query_params=['expand'], relative_path='v1/{+parent}/reports', request_field='', request_type_name='ApigeeOrganizationsReportsListRequest', response_type_name='GoogleCloudApigeeV1ListCustomReportsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Update an existing custom report definition.

      Args:
        request: (GoogleCloudApigeeV1CustomReport) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1CustomReport) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/reports/{reportsId}', http_method='PUT', method_id='apigee.organizations.reports.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1CustomReport', response_type_name='GoogleCloudApigeeV1CustomReport', supports_download=False)