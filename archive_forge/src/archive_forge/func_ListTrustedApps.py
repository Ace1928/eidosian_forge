from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
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