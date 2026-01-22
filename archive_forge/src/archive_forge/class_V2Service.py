from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class V2Service(base_api.BaseApiService):
    """Service class for the v2 resource."""
    _NAME = 'v2'

    def __init__(self, client):
        super(LoggingV2.V2Service, self).__init__(client)
        self._upload_configs = {}

    def GetCmekSettings(self, request, global_params=None):
        """Gets the Logging CMEK settings for the given resource.Note: CMEK for the Log Router can be configured for Google Cloud projects, folders, organizations, and billing accounts. Once configured for an organization, it applies to all projects and folders in the Google Cloud organization.See Enabling CMEK for Log Router (https://cloud.google.com/logging/docs/routing/managed-encryption) for more information.

      Args:
        request: (LoggingGetCmekSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CmekSettings) The response message.
      """
        config = self.GetMethodConfig('GetCmekSettings')
        return self._RunMethod(config, request, global_params=global_params)
    GetCmekSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/cmekSettings', http_method='GET', method_id='logging.getCmekSettings', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}/cmekSettings', request_field='', request_type_name='LoggingGetCmekSettingsRequest', response_type_name='CmekSettings', supports_download=False)

    def GetSettings(self, request, global_params=None):
        """Gets the settings for the given resource.Note: Settings can be retrieved for Google Cloud projects, folders, organizations, and billing accounts.See View default resource settings for Logging (https://cloud.google.com/logging/docs/default-settings#view-org-settings) for more information.

      Args:
        request: (LoggingGetSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Settings) The response message.
      """
        config = self.GetMethodConfig('GetSettings')
        return self._RunMethod(config, request, global_params=global_params)
    GetSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/settings', http_method='GET', method_id='logging.getSettings', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}/settings', request_field='', request_type_name='LoggingGetSettingsRequest', response_type_name='Settings', supports_download=False)

    def UpdateCmekSettings(self, request, global_params=None):
        """Updates the Log Router CMEK settings for the given resource.Note: CMEK for the Log Router can currently only be configured for Google Cloud organizations. Once configured, it applies to all projects and folders in the Google Cloud organization.UpdateCmekSettings fails when any of the following are true: The value of kms_key_name is invalid. The associated service account doesn't have the required roles/cloudkms.cryptoKeyEncrypterDecrypter role assigned for the key. Access to the key is disabled.See Enabling CMEK for Log Router (https://cloud.google.com/logging/docs/routing/managed-encryption) for more information.

      Args:
        request: (LoggingUpdateCmekSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CmekSettings) The response message.
      """
        config = self.GetMethodConfig('UpdateCmekSettings')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateCmekSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/cmekSettings', http_method='PATCH', method_id='logging.updateCmekSettings', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}/cmekSettings', request_field='cmekSettings', request_type_name='LoggingUpdateCmekSettingsRequest', response_type_name='CmekSettings', supports_download=False)

    def UpdateSettings(self, request, global_params=None):
        """Updates the settings for the given resource. This method applies to all feature configurations for organization and folders.UpdateSettings fails when any of the following are true: The value of storage_location either isn't supported by Logging or violates the location OrgPolicy. The default_sink_config field is set, but it has an unspecified filter write mode. The value of kms_key_name is invalid. The associated service account doesn't have the required roles/cloudkms.cryptoKeyEncrypterDecrypter role assigned for the key. Access to the key is disabled.See Configure default settings for organizations and folders (https://cloud.google.com/logging/docs/default-settings) for more information.

      Args:
        request: (LoggingUpdateSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Settings) The response message.
      """
        config = self.GetMethodConfig('UpdateSettings')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/settings', http_method='PATCH', method_id='logging.updateSettings', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}/settings', request_field='settings', request_type_name='LoggingUpdateSettingsRequest', response_type_name='Settings', supports_download=False)