from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v1 import securitycenter_v1_messages as messages
class OrganizationsBigQueryExportsService(base_api.BaseApiService):
    """Service class for the organizations_bigQueryExports resource."""
    _NAME = 'organizations_bigQueryExports'

    def __init__(self, client):
        super(SecuritycenterV1.OrganizationsBigQueryExportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a BigQuery export.

      Args:
        request: (SecuritycenterOrganizationsBigQueryExportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1BigQueryExport) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/bigQueryExports', http_method='POST', method_id='securitycenter.organizations.bigQueryExports.create', ordered_params=['parent'], path_params=['parent'], query_params=['bigQueryExportId'], relative_path='v1/{+parent}/bigQueryExports', request_field='googleCloudSecuritycenterV1BigQueryExport', request_type_name='SecuritycenterOrganizationsBigQueryExportsCreateRequest', response_type_name='GoogleCloudSecuritycenterV1BigQueryExport', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing BigQuery export.

      Args:
        request: (SecuritycenterOrganizationsBigQueryExportsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/bigQueryExports/{bigQueryExportsId}', http_method='DELETE', method_id='securitycenter.organizations.bigQueryExports.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsBigQueryExportsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a BigQuery export.

      Args:
        request: (SecuritycenterOrganizationsBigQueryExportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1BigQueryExport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/bigQueryExports/{bigQueryExportsId}', http_method='GET', method_id='securitycenter.organizations.bigQueryExports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SecuritycenterOrganizationsBigQueryExportsGetRequest', response_type_name='GoogleCloudSecuritycenterV1BigQueryExport', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BigQuery exports. Note that when requesting BigQuery exports at a given level all exports under that level are also returned e.g. if requesting BigQuery exports under a folder, then all BigQuery exports immediately under the folder plus the ones created under the projects within the folder are returned.

      Args:
        request: (SecuritycenterOrganizationsBigQueryExportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBigQueryExportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/bigQueryExports', http_method='GET', method_id='securitycenter.organizations.bigQueryExports.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/bigQueryExports', request_field='', request_type_name='SecuritycenterOrganizationsBigQueryExportsListRequest', response_type_name='ListBigQueryExportsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a BigQuery export.

      Args:
        request: (SecuritycenterOrganizationsBigQueryExportsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV1BigQueryExport) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/bigQueryExports/{bigQueryExportsId}', http_method='PATCH', method_id='securitycenter.organizations.bigQueryExports.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudSecuritycenterV1BigQueryExport', request_type_name='SecuritycenterOrganizationsBigQueryExportsPatchRequest', response_type_name='GoogleCloudSecuritycenterV1BigQueryExport', supports_download=False)