from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersAppsKeysService(base_api.BaseApiService):
    """Service class for the organizations_developers_apps_keys resource."""
    _NAME = 'organizations_developers_apps_keys'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersAppsKeysService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a custom consumer key and secret for a developer app. This is particularly useful if you want to migrate existing consumer keys and secrets to Apigee from another system. Consumer keys and secrets can contain letters, numbers, underscores, and hyphens. No other special characters are allowed. To avoid service disruptions, a consumer key and secret should not exceed 2 KBs each. **Note**: When creating the consumer key and secret, an association to API products will not be made. Therefore, you should not specify the associated API products in your request. Instead, use the UpdateDeveloperAppKey API to make the association after the consumer key and secret are created. If a consumer key and secret already exist, you can keep them or delete them using the DeleteDeveloperAppKey API. **Note**: All keys start out with status=approved, even if status=revoked is passed when the key is created. To revoke a key, use the UpdateDeveloperAppKey API.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys', http_method='POST', method_id='apigee.organizations.developers.apps.keys.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/keys', request_field='googleCloudApigeeV1DeveloperAppKey', request_type_name='ApigeeOrganizationsDevelopersAppsKeysCreateRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an app's consumer key and removes all API products associated with the app. After the consumer key is deleted, it cannot be used to access any APIs. **Note**: After you delete a consumer key, you may want to: 1. Create a new consumer key and secret for the developer app using the CreateDeveloperAppKey API, and subsequently add an API product to the key using the UpdateDeveloperAppKey API. 2. Delete the developer app, if it is no longer required.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/{keysId}', http_method='DELETE', method_id='apigee.organizations.developers.apps.keys.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsKeysDeleteRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details for a consumer key for a developer app, including the key and secret value, associated API products, and other information.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/{keysId}', http_method='GET', method_id='apigee.organizations.developers.apps.keys.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersAppsKeysGetRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)

    def ReplaceDeveloperAppKey(self, request, global_params=None):
        """Updates the scope of an app. This API replaces the existing scopes with those specified in the request. Include or exclude any existing scopes that you want to retain or delete, respectively. The specified scopes must already be defined for the API products associated with the app. This API sets the `scopes` element under the `apiProducts` element in the attributes of the app.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysReplaceDeveloperAppKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('ReplaceDeveloperAppKey')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceDeveloperAppKey.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/{keysId}', http_method='PUT', method_id='apigee.organizations.developers.apps.keys.replaceDeveloperAppKey', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1DeveloperAppKey', request_type_name='ApigeeOrganizationsDevelopersAppsKeysReplaceDeveloperAppKeyRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)

    def UpdateDeveloperAppKey(self, request, global_params=None):
        """Adds an API product to a developer app key, enabling the app that holds the key to access the API resources bundled in the API product. In addition, you can add attributes to a developer app key. This API replaces the existing attributes with those specified in the request. Include or exclude any existing attributes that you want to retain or delete, respectively. You can use the same key to access all API products associated with the app.

      Args:
        request: (ApigeeOrganizationsDevelopersAppsKeysUpdateDeveloperAppKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperAppKey) The response message.
      """
        config = self.GetMethodConfig('UpdateDeveloperAppKey')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDeveloperAppKey.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/apps/{appsId}/keys/{keysId}', http_method='POST', method_id='apigee.organizations.developers.apps.keys.updateDeveloperAppKey', ordered_params=['name'], path_params=['name'], query_params=['action'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1DeveloperAppKey', request_type_name='ApigeeOrganizationsDevelopersAppsKeysUpdateDeveloperAppKeyRequest', response_type_name='GoogleCloudApigeeV1DeveloperAppKey', supports_download=False)