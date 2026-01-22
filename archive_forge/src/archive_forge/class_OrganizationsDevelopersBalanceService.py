from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersBalanceService(base_api.BaseApiService):
    """Service class for the organizations_developers_balance resource."""
    _NAME = 'organizations_developers_balance'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersBalanceService, self).__init__(client)
        self._upload_configs = {}

    def Adjust(self, request, global_params=None):
        """Adjust the prepaid balance for the developer. This API will be used in scenarios where the developer has been under-charged or over-charged.

      Args:
        request: (ApigeeOrganizationsDevelopersBalanceAdjustRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperBalance) The response message.
      """
        config = self.GetMethodConfig('Adjust')
        return self._RunMethod(config, request, global_params=global_params)
    Adjust.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/balance:adjust', http_method='POST', method_id='apigee.organizations.developers.balance.adjust', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:adjust', request_field='googleCloudApigeeV1AdjustDeveloperBalanceRequest', request_type_name='ApigeeOrganizationsDevelopersBalanceAdjustRequest', response_type_name='GoogleCloudApigeeV1DeveloperBalance', supports_download=False)

    def Credit(self, request, global_params=None):
        """Credits the account balance for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersBalanceCreditRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperBalance) The response message.
      """
        config = self.GetMethodConfig('Credit')
        return self._RunMethod(config, request, global_params=global_params)
    Credit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/balance:credit', http_method='POST', method_id='apigee.organizations.developers.balance.credit', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:credit', request_field='googleCloudApigeeV1CreditDeveloperBalanceRequest', request_type_name='ApigeeOrganizationsDevelopersBalanceCreditRequest', response_type_name='GoogleCloudApigeeV1DeveloperBalance', supports_download=False)