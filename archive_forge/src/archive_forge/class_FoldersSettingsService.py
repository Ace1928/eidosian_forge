from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.resourcesettings.v1 import resourcesettings_v1_messages as messages
class FoldersSettingsService(base_api.BaseApiService):
    """Service class for the folders_settings resource."""
    _NAME = 'folders_settings'

    def __init__(self, client):
        super(ResourcesettingsV1.FoldersSettingsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns a specified setting. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the setting does not exist.

      Args:
        request: (ResourcesettingsFoldersSettingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudResourcesettingsV1Setting) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/settings/{settingsId}', http_method='GET', method_id='resourcesettings.folders.settings.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='ResourcesettingsFoldersSettingsGetRequest', response_type_name='GoogleCloudResourcesettingsV1Setting', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the settings that are available on the Cloud resource `parent`.

      Args:
        request: (ResourcesettingsFoldersSettingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudResourcesettingsV1ListSettingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/settings', http_method='GET', method_id='resourcesettings.folders.settings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/settings', request_field='', request_type_name='ResourcesettingsFoldersSettingsListRequest', response_type_name='GoogleCloudResourcesettingsV1ListSettingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a specified setting. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the setting does not exist. Returns a `google.rpc.Status` with `google.rpc.Code.FAILED_PRECONDITION` if the setting is flagged as read only. Returns a `google.rpc.Status` with `google.rpc.Code.ABORTED` if the etag supplied in the request does not match the persisted etag of the setting value. On success, the response will contain only `name`, `local_value` and `etag`. The `metadata` and `effective_value` cannot be updated through this API. Note: the supplied setting will perform a full overwrite of the `local_value` field.

      Args:
        request: (ResourcesettingsFoldersSettingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudResourcesettingsV1Setting) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/folders/{foldersId}/settings/{settingsId}', http_method='PATCH', method_id='resourcesettings.folders.settings.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='googleCloudResourcesettingsV1Setting', request_type_name='ResourcesettingsFoldersSettingsPatchRequest', response_type_name='GoogleCloudResourcesettingsV1Setting', supports_download=False)