from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.resourcesettings.v1alpha1 import resourcesettings_v1alpha1_messages as messages
class OrganizationsSettingsValueService(base_api.BaseApiService):
    """Service class for the organizations_settings_value resource."""
    _NAME = 'organizations_settings_value'

    def __init__(self, client):
        super(ResourcesettingsV1alpha1.OrganizationsSettingsValueService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a setting value. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the setting does not exist. Returns a `google.rpc.Status` with `google.rpc.Code.ALREADY_EXISTS` if the setting value already exists on the given Cloud resource. Returns a `google.rpc.Status` with `google.rpc.Code.FAILED_PRECONDITION` if the setting is flagged as read only.

      Args:
        request: (ResourcesettingsOrganizationsSettingsValueCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudResourcesettingsV1alpha1SettingValue) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/organizations/{organizationsId}/settings/{settingsId}/value', http_method='POST', method_id='resourcesettings.organizations.settings.value.create', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='googleCloudResourcesettingsV1alpha1SettingValue', request_type_name='ResourcesettingsOrganizationsSettingsValueCreateRequest', response_type_name='GoogleCloudResourcesettingsV1alpha1SettingValue', supports_download=False)