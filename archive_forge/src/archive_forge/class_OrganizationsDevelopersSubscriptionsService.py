from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersSubscriptionsService(base_api.BaseApiService):
    """Service class for the organizations_developers_subscriptions resource."""
    _NAME = 'organizations_developers_subscriptions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersSubscriptionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a subscription to an API product. .

      Args:
        request: (ApigeeOrganizationsDevelopersSubscriptionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperSubscription) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/subscriptions', http_method='POST', method_id='apigee.organizations.developers.subscriptions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/subscriptions', request_field='googleCloudApigeeV1DeveloperSubscription', request_type_name='ApigeeOrganizationsDevelopersSubscriptionsCreateRequest', response_type_name='GoogleCloudApigeeV1DeveloperSubscription', supports_download=False)

    def Expire(self, request, global_params=None):
        """Expires an API product subscription immediately.

      Args:
        request: (ApigeeOrganizationsDevelopersSubscriptionsExpireRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperSubscription) The response message.
      """
        config = self.GetMethodConfig('Expire')
        return self._RunMethod(config, request, global_params=global_params)
    Expire.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/subscriptions/{subscriptionsId}:expire', http_method='POST', method_id='apigee.organizations.developers.subscriptions.expire', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:expire', request_field='googleCloudApigeeV1ExpireDeveloperSubscriptionRequest', request_type_name='ApigeeOrganizationsDevelopersSubscriptionsExpireRequest', response_type_name='GoogleCloudApigeeV1DeveloperSubscription', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details for an API product subscription.

      Args:
        request: (ApigeeOrganizationsDevelopersSubscriptionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperSubscription) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/subscriptions/{subscriptionsId}', http_method='GET', method_id='apigee.organizations.developers.subscriptions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersSubscriptionsGetRequest', response_type_name='GoogleCloudApigeeV1DeveloperSubscription', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all API product subscriptions for a developer.

      Args:
        request: (ApigeeOrganizationsDevelopersSubscriptionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDeveloperSubscriptionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/subscriptions', http_method='GET', method_id='apigee.organizations.developers.subscriptions.list', ordered_params=['parent'], path_params=['parent'], query_params=['count', 'startKey'], relative_path='v1/{+parent}/subscriptions', request_field='', request_type_name='ApigeeOrganizationsDevelopersSubscriptionsListRequest', response_type_name='GoogleCloudApigeeV1ListDeveloperSubscriptionsResponse', supports_download=False)