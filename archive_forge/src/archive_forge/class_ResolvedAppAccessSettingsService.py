from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class ResolvedAppAccessSettingsService(base_api.BaseApiService):
    """Service class for the resolvedAppAccessSettings resource."""
    _NAME = u'resolvedAppAccessSettings'

    def __init__(self, client):
        super(AdminDirectoryV1.ResolvedAppAccessSettingsService, self).__init__(client)
        self._upload_configs = {}

    def GetSettings(self, request, global_params=None):
        """Retrieves resolved app access settings of the logged in user.

      Args:
        request: (DirectoryResolvedAppAccessSettingsGetSettingsRequest) input
          message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (AppAccessCollections) The response message.
      """
        config = self.GetMethodConfig('GetSettings')
        return self._RunMethod(config, request, global_params=global_params)
    GetSettings.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resolvedAppAccessSettings.GetSettings', ordered_params=[], path_params=[], query_params=[], relative_path=u'resolvedappaccesssettings', request_field='', request_type_name=u'DirectoryResolvedAppAccessSettingsGetSettingsRequest', response_type_name=u'AppAccessCollections', supports_download=False)

    def ListTrustedApps(self, request, global_params=None):
        """Retrieves the list of apps trusted by the admin of the logged in user.

      Args:
        request: (DirectoryResolvedAppAccessSettingsListTrustedAppsRequest)
          input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (TrustedApps) The response message.
      """
        config = self.GetMethodConfig('ListTrustedApps')
        return self._RunMethod(config, request, global_params=global_params)
    ListTrustedApps.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.resolvedAppAccessSettings.ListTrustedApps', ordered_params=[], path_params=[], query_params=[], relative_path=u'trustedapps', request_field='', request_type_name=u'DirectoryResolvedAppAccessSettingsListTrustedAppsRequest', response_type_name=u'TrustedApps', supports_download=False)