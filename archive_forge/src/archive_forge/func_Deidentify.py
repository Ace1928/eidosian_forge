from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
def Deidentify(self, request, global_params=None):
    """De-identifies potentially sensitive info from a ContentItem. This method has limits on input size and output size. See https://cloud.google.com/sensitive-data-protection/docs/deidentify-sensitive-data to learn more. When no InfoTypes or CustomInfoTypes are specified in this request, the system will automatically choose what detectors to run. By default this may be all types, but may change over time as detectors are updated.

      Args:
        request: (DlpProjectsLocationsContentDeidentifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DeidentifyContentResponse) The response message.
      """
    config = self.GetMethodConfig('Deidentify')
    return self._RunMethod(config, request, global_params=global_params)