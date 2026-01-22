from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as messages
def GetProjectFeatureSettings(self, request, global_params=None):
    """GetProjectFeatureSettings returns the feature settings for a project.

      Args:
        request: (OsconfigProjectsLocationsGlobalGetProjectFeatureSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectFeatureSettings) The response message.
      """
    config = self.GetMethodConfig('GetProjectFeatureSettings')
    return self._RunMethod(config, request, global_params=global_params)