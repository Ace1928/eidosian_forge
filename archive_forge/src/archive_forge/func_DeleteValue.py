from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.resourcesettings.v1alpha1 import resourcesettings_v1alpha1_messages as messages
def DeleteValue(self, request, global_params=None):
    """Deletes a setting value. If the setting value does not exist, the operation is a no-op. Returns a `google.rpc.Status` with `google.rpc.Code.NOT_FOUND` if the setting or the setting value does not exist. The setting value will not exist if a prior call to `DeleteSetting` for the setting value already returned a success code. Returns a `google.rpc.Status` with `google.rpc.Code.FAILED_PRECONDITION` if the setting is flagged as read only.

      Args:
        request: (ResourcesettingsProjectsSettingsDeleteValueRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('DeleteValue')
    return self._RunMethod(config, request, global_params=global_params)