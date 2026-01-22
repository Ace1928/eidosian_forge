from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsDevelopersService(base_api.BaseApiService):
    """Service class for the organizations_developers resource."""
    _NAME = 'organizations_developers'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsDevelopersService, self).__init__(client)
        self._upload_configs = {}

    def Attributes(self, request, global_params=None):
        """Updates developer attributes. This API replaces the existing attributes with those specified in the request. Add new attributes, and include or exclude any existing attributes that you want to retain or remove, respectively. The custom attribute limit is 18. **Note**: OAuth access tokens and Key Management Service (KMS) entities (apps, developers, and API products) are cached for 180 seconds (default). Any custom attributes associated with these entities are cached for at least 180 seconds after the entity is accessed at runtime. Therefore, an `ExpiresIn` element on the OAuthV2 policy won't be able to expire an access token in less than 180 seconds.

      Args:
        request: (ApigeeOrganizationsDevelopersAttributesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Attributes) The response message.
      """
        config = self.GetMethodConfig('Attributes')
        return self._RunMethod(config, request, global_params=global_params)
    Attributes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/attributes', http_method='POST', method_id='apigee.organizations.developers.attributes', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/attributes', request_field='googleCloudApigeeV1Attributes', request_type_name='ApigeeOrganizationsDevelopersAttributesRequest', response_type_name='GoogleCloudApigeeV1Attributes', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a developer. Once created, the developer can register an app and obtain an API key. At creation time, a developer is set as `active`. To change the developer status, use the SetDeveloperStatus API.

      Args:
        request: (ApigeeOrganizationsDevelopersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Developer) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers', http_method='POST', method_id='apigee.organizations.developers.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/developers', request_field='googleCloudApigeeV1Developer', request_type_name='ApigeeOrganizationsDevelopersCreateRequest', response_type_name='GoogleCloudApigeeV1Developer', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a developer. All apps and API keys associated with the developer are also removed. **Warning**: This API will permanently delete the developer and related artifacts. To avoid permanently deleting developers and their artifacts, set the developer status to `inactive` using the SetDeveloperStatus API. **Note**: The delete operation is asynchronous. The developer app is deleted immediately, but its associated resources, such as apps and API keys, may take anywhere from a few seconds to a few minutes to be deleted.

      Args:
        request: (ApigeeOrganizationsDevelopersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Developer) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}', http_method='DELETE', method_id='apigee.organizations.developers.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersDeleteRequest', response_type_name='GoogleCloudApigeeV1Developer', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the developer details, including the developer's name, email address, apps, and other information. **Note**: The response includes only the first 100 developer apps.

      Args:
        request: (ApigeeOrganizationsDevelopersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Developer) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}', http_method='GET', method_id='apigee.organizations.developers.get', ordered_params=['name'], path_params=['name'], query_params=['action'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersGetRequest', response_type_name='GoogleCloudApigeeV1Developer', supports_download=False)

    def GetBalance(self, request, global_params=None):
        """Gets the account balance for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersGetBalanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperBalance) The response message.
      """
        config = self.GetMethodConfig('GetBalance')
        return self._RunMethod(config, request, global_params=global_params)
    GetBalance.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/balance', http_method='GET', method_id='apigee.organizations.developers.getBalance', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersGetBalanceRequest', response_type_name='GoogleCloudApigeeV1DeveloperBalance', supports_download=False)

    def GetMonetizationConfig(self, request, global_params=None):
        """Gets the monetization configuration for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersGetMonetizationConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperMonetizationConfig) The response message.
      """
        config = self.GetMethodConfig('GetMonetizationConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetMonetizationConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/monetizationConfig', http_method='GET', method_id='apigee.organizations.developers.getMonetizationConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersGetMonetizationConfigRequest', response_type_name='GoogleCloudApigeeV1DeveloperMonetizationConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all developers in an organization by email address. By default, the response does not include company developers. Set the `includeCompany` query parameter to `true` to include company developers. **Note**: A maximum of 1000 developers are returned in the response. You paginate the list of developers returned using the `startKey` and `count` query parameters.

      Args:
        request: (ApigeeOrganizationsDevelopersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListOfDevelopersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers', http_method='GET', method_id='apigee.organizations.developers.list', ordered_params=['parent'], path_params=['parent'], query_params=['app', 'count', 'expand', 'filter', 'ids', 'includeCompany', 'pageSize', 'pageToken', 'startKey'], relative_path='v1/{+parent}/developers', request_field='', request_type_name='ApigeeOrganizationsDevelopersListRequest', response_type_name='GoogleCloudApigeeV1ListOfDevelopersResponse', supports_download=False)

    def SetDeveloperStatus(self, request, global_params=None):
        """Sets the status of a developer. A developer is `active` by default. If you set a developer's status to `inactive`, the API keys assigned to the developer apps are no longer valid even though the API keys are set to `approved`. Inactive developers can still sign in to the developer portal and create apps; however, any new API keys generated during app creation won't work. To set the status of a developer, set the `action` query parameter to `active` or `inactive`, and the `Content-Type` header to `application/octet-stream`. If successful, the API call returns the following HTTP status code: `204 No Content`.

      Args:
        request: (ApigeeOrganizationsDevelopersSetDeveloperStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('SetDeveloperStatus')
        return self._RunMethod(config, request, global_params=global_params)
    SetDeveloperStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}', http_method='POST', method_id='apigee.organizations.developers.setDeveloperStatus', ordered_params=['name'], path_params=['name'], query_params=['action'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsDevelopersSetDeveloperStatusRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a developer. This API replaces the existing developer details with those specified in the request. Include or exclude any existing details that you want to retain or delete, respectively. The custom attribute limit is 18. **Note**: OAuth access tokens and Key Management Service (KMS) entities (apps, developers, and API products) are cached for 180 seconds (current default). Any custom attributes associated with these entities are cached for at least 180 seconds after the entity is accessed at runtime. Therefore, an `ExpiresIn` element on the OAuthV2 policy won't be able to expire an access token in less than 180 seconds.

      Args:
        request: (ApigeeOrganizationsDevelopersUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Developer) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}', http_method='PUT', method_id='apigee.organizations.developers.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1Developer', request_type_name='ApigeeOrganizationsDevelopersUpdateRequest', response_type_name='GoogleCloudApigeeV1Developer', supports_download=False)

    def UpdateMonetizationConfig(self, request, global_params=None):
        """Updates the monetization configuration for the developer.

      Args:
        request: (ApigeeOrganizationsDevelopersUpdateMonetizationConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DeveloperMonetizationConfig) The response message.
      """
        config = self.GetMethodConfig('UpdateMonetizationConfig')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateMonetizationConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/developers/{developersId}/monetizationConfig', http_method='PUT', method_id='apigee.organizations.developers.updateMonetizationConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudApigeeV1DeveloperMonetizationConfig', request_type_name='ApigeeOrganizationsDevelopersUpdateMonetizationConfigRequest', response_type_name='GoogleCloudApigeeV1DeveloperMonetizationConfig', supports_download=False)