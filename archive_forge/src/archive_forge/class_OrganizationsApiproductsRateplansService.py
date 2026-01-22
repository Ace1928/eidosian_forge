from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsApiproductsRateplansService(base_api.BaseApiService):
    """Service class for the organizations_apiproducts_rateplans resource."""
    _NAME = 'organizations_apiproducts_rateplans'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsApiproductsRateplansService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a rate plan that is associated with an API product in an organization. Using rate plans, API product owners can monetize their API products by configuring one or more of the following: - Billing frequency - Initial setup fees for using an API product - Payment funding model (postpaid only) - Fixed recurring or consumption-based charges for using an API product - Revenue sharing with developer partners An API product can have multiple rate plans associated with it but *only one* rate plan can be active at any point of time. **Note: From the developer's perspective, they purchase API products not rate plans.

      Args:
        request: (ApigeeOrganizationsApiproductsRateplansCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1RatePlan) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/rateplans', http_method='POST', method_id='apigee.organizations.apiproducts.rateplans.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/rateplans', request_field='googleCloudApigeeV1RatePlan', request_type_name='ApigeeOrganizationsApiproductsRateplansCreateRequest', response_type_name='GoogleCloudApigeeV1RatePlan', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a rate plan.

      Args:
        request: (ApigeeOrganizationsApiproductsRateplansDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1RatePlan) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/rateplans/{rateplansId}', http_method='DELETE', method_id='apigee.organizations.apiproducts.rateplans.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApiproductsRateplansDeleteRequest', response_type_name='GoogleCloudApigeeV1RatePlan', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a rate plan.

      Args:
        request: (ApigeeOrganizationsApiproductsRateplansGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1RatePlan) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/rateplans/{rateplansId}', http_method='GET', method_id='apigee.organizations.apiproducts.rateplans.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsApiproductsRateplansGetRequest', response_type_name='GoogleCloudApigeeV1RatePlan', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the rate plans for an API product.

      Args:
        request: (ApigeeOrganizationsApiproductsRateplansListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListRatePlansResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/rateplans', http_method='GET', method_id='apigee.organizations.apiproducts.rateplans.list', ordered_params=['parent'], path_params=['parent'], query_params=['count', 'expand', 'orderBy', 'startKey', 'state'], relative_path='v1/{+parent}/rateplans', request_field='', request_type_name='ApigeeOrganizationsApiproductsRateplansListRequest', response_type_name='GoogleCloudApigeeV1ListRatePlansResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing rate plan.

      Args:
        request: (GoogleCloudApigeeV1RatePlan) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1RatePlan) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/apiproducts/{apiproductsId}/rateplans/{rateplansId}', http_method='PUT', method_id='apigee.organizations.apiproducts.rateplans.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1RatePlan', response_type_name='GoogleCloudApigeeV1RatePlan', supports_download=False)