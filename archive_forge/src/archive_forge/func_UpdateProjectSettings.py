from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
def UpdateProjectSettings(self, request, global_params=None):
    """Updates the Settings for the Project.

      Args:
        request: (ArtifactregistryProjectsUpdateProjectSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateProjectSettings')
    return self._RunMethod(config, request, global_params=global_params)