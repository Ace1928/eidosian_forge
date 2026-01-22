from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
def ExportImage(self, request, global_params=None):
    """Export image for a given resource.

      Args:
        request: (RunProjectsLocationsExportImageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ExportImageResponse) The response message.
      """
    config = self.GetMethodConfig('ExportImage')
    return self._RunMethod(config, request, global_params=global_params)