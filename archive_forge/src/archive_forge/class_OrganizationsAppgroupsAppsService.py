from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsAppgroupsAppsService(base_api.BaseApiService):
    """Service class for the organizations_appgroups_apps resource."""
    _NAME = 'organizations_appgroups_apps'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsAppgroupsAppsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an app and associates it with an AppGroup. This API associates the AppGroup app with the specified API product and auto-generates an API key for the app to use in calls to API proxies inside that API product. The `name` is the unique ID of the app that you can use in API calls.

      Args:
        request: (ApigeeOrganizationsAppgroupsAppsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AppGroupApp) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/appgroups/{appgroupsId}/apps', http_method='POST', method_id='apigee.organizations.appgroups.apps.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/apps', request_field='googleCloudApigeeV1AppGroupApp', request_type_name='ApigeeOrganizationsAppgroupsAppsCreateRequest', response_type_name='GoogleCloudApigeeV1AppGroupApp', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an AppGroup app. **Note**: The delete operation is asynchronous. The AppGroup app is deleted immediately, but its associated resources, such as app keys or access tokens, may take anywhere from a few seconds to a few minutes to be deleted.

      Args:
        request: (ApigeeOrganizationsAppgroupsAppsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AppGroupApp) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/appgroups/{appgroupsId}/apps/{appsId}', http_method='DELETE', method_id='apigee.organizations.appgroups.apps.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsAppgroupsAppsDeleteRequest', response_type_name='GoogleCloudApigeeV1AppGroupApp', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the details for an AppGroup app.

      Args:
        request: (ApigeeOrganizationsAppgroupsAppsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AppGroupApp) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/appgroups/{appgroupsId}/apps/{appsId}', http_method='GET', method_id='apigee.organizations.appgroups.apps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsAppgroupsAppsGetRequest', response_type_name='GoogleCloudApigeeV1AppGroupApp', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all apps created by an AppGroup in an Apigee organization. Optionally, you can request an expanded view of the AppGroup apps. Lists all AppGroupApps in an AppGroup. A maximum of 1000 AppGroup apps are returned in the response if PageSize is not specified, or if the PageSize is greater than 1000.

      Args:
        request: (ApigeeOrganizationsAppgroupsAppsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListAppGroupAppsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/appgroups/{appgroupsId}/apps', http_method='GET', method_id='apigee.organizations.appgroups.apps.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/apps', request_field='', request_type_name='ApigeeOrganizationsAppgroupsAppsListRequest', response_type_name='GoogleCloudApigeeV1ListAppGroupAppsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the details for an AppGroup app. In addition, you can add an API product to an AppGroup app and automatically generate an API key for the app to use when calling APIs in the API product. If you want to use an existing API key for the API product, add the API product to the API key using the UpdateAppGroupAppKey API. Using this API, you cannot update the app name, as it is the primary key used to identify the app and cannot be changed. This API replaces the existing attributes with those specified in the request. Include or exclude any existing attributes that you want to retain or delete, respectively.

      Args:
        request: (ApigeeOrganizationsAppgroupsAppsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AppGroupApp) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/appgroups/{appgroupsId}/apps/{appsId}', http_method='PUT', method_id='apigee.organizations.appgroups.apps.update', ordered_params=['name'], path_params=['name'], query_params=['action'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1AppGroupApp', request_type_name='ApigeeOrganizationsAppgroupsAppsUpdateRequest', response_type_name='GoogleCloudApigeeV1AppGroupApp', supports_download=False)