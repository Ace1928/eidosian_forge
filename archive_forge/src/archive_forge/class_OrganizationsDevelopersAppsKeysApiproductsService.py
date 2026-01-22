from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersAppsKeysApiproductsService(base_api.BaseApiService):
    """Service class for the organizations_developers_apps_keys_apiproducts resource."""
    _NAME = 'organizations_developers_apps_keys_apiproducts'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersAppsKeysApiproductsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Removes an API product from an app's consumer key. After the API product is removed, the app cannot access the API resources defined in that API product. **Note**: The consumer key is not removed, only its association with the API product.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysApiproductsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/{keysId}/apiproducts/{apiproductsId}', http_method='DELETE', method_id='apigee.organizations.developers.apps.keys.apiproducts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsKeysApiproductsDeleteRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)

    def UpdateDeveloperAppKeyApiProduct(self, request, global_params=None):
        """Approves or revokes the consumer key for an API product. After a consumer key is approved, the app can use it to access APIs. A consumer key that is revoked or pending cannot be used to access an API. Any access tokens associated with a revoked consumer key will remain active. However, Apigee checks the status of the consumer key and if set to `revoked` will not allow access to the API.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysApiproductsUpdateDeveloperAppKeyApiProductRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('UpdateDeveloperAppKeyApiProduct')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDeveloperAppKeyApiProduct.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/{keysId}/apiproducts/{apiproductsId}', http_method='POST', method_id='apigee.organizations.developers.apps.keys.apiproducts.updateDeveloperAppKeyApiProduct', ordered_params=['name'], path_params=['name'], query_params=['action'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsKeysApiproductsUpdateDeveloperAppKeyApiProductRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)