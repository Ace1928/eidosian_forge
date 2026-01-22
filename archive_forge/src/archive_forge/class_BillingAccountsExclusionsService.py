from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class BillingAccountsExclusionsService(base_api.BaseApiService):
    """Service class for the billingAccounts_exclusions resource."""
    _NAME = 'billingAccounts_exclusions'

    def __init__(self, client):
        super(LoggingV2.BillingAccountsExclusionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new exclusion in the _Default sink in a specified parent resource. Only log entries belonging to that resource can be excluded. You can have up to 10 exclusions in a resource.

      Args:
        request: (LoggingBillingAccountsExclusionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogExclusion) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/exclusions', http_method='POST', method_id='logging.billingAccounts.exclusions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/exclusions', request_field='logExclusion', request_type_name='LoggingBillingAccountsExclusionsCreateRequest', response_type_name='LogExclusion', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an exclusion in the _Default sink.

      Args:
        request: (LoggingBillingAccountsExclusionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/exclusions/{exclusionsId}', http_method='DELETE', method_id='logging.billingAccounts.exclusions.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingBillingAccountsExclusionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the description of an exclusion in the _Default sink.

      Args:
        request: (LoggingBillingAccountsExclusionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogExclusion) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/exclusions/{exclusionsId}', http_method='GET', method_id='logging.billingAccounts.exclusions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingBillingAccountsExclusionsGetRequest', response_type_name='LogExclusion', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the exclusions on the _Default sink in a parent resource.

      Args:
        request: (LoggingBillingAccountsExclusionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExclusionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/exclusions', http_method='GET', method_id='logging.billingAccounts.exclusions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/exclusions', request_field='', request_type_name='LoggingBillingAccountsExclusionsListRequest', response_type_name='ListExclusionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Changes one or more properties of an existing exclusion in the _Default sink.

      Args:
        request: (LoggingBillingAccountsExclusionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogExclusion) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/billingAccounts/{billingAccountsId}/exclusions/{exclusionsId}', http_method='PATCH', method_id='logging.billingAccounts.exclusions.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='logExclusion', request_type_name='LoggingBillingAccountsExclusionsPatchRequest', response_type_name='LogExclusion', supports_download=False)