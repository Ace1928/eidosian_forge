from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
def Reidentify(self, request, global_params=None):
    """Re-identifies content that has been de-identified. See https://cloud.google.com/sensitive-data-protection/docs/pseudonymization#re-identification_in_free_text_code_example to learn more.

      Args:
        request: (DlpProjectsLocationsContentReidentifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ReidentifyContentResponse) The response message.
      """
    config = self.GetMethodConfig('Reidentify')
    return self._RunMethod(config, request, global_params=global_params)