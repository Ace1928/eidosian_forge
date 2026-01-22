from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsContentService(base_api.BaseApiService):
    """Service class for the projects_content resource."""
    _NAME = 'projects_content'

    def __init__(self, client):
        super(DlpV2.ProjectsContentService, self).__init__(client)
        self._upload_configs = {}

    def Deidentify(self, request, global_params=None):
        """De-identifies potentially sensitive info from a ContentItem. This method has limits on input size and output size. See https://cloud.google.com/sensitive-data-protection/docs/deidentify-sensitive-data to learn more. When no InfoTypes or CustomInfoTypes are specified in this request, the system will automatically choose what detectors to run. By default this may be all types, but may change over time as detectors are updated.

      Args:
        request: (DlpProjectsContentDeidentifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DeidentifyContentResponse) The response message.
      """
        config = self.GetMethodConfig('Deidentify')
        return self._RunMethod(config, request, global_params=global_params)
    Deidentify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/content:deidentify', http_method='POST', method_id='dlp.projects.content.deidentify', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/content:deidentify', request_field='googlePrivacyDlpV2DeidentifyContentRequest', request_type_name='DlpProjectsContentDeidentifyRequest', response_type_name='GooglePrivacyDlpV2DeidentifyContentResponse', supports_download=False)

    def Inspect(self, request, global_params=None):
        """Finds potentially sensitive info in content. This method has limits on input size, processing time, and output size. When no InfoTypes or CustomInfoTypes are specified in this request, the system will automatically choose what detectors to run. By default this may be all types, but may change over time as detectors are updated. For how to guides, see https://cloud.google.com/sensitive-data-protection/docs/inspecting-images and https://cloud.google.com/sensitive-data-protection/docs/inspecting-text,.

      Args:
        request: (DlpProjectsContentInspectRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2InspectContentResponse) The response message.
      """
        config = self.GetMethodConfig('Inspect')
        return self._RunMethod(config, request, global_params=global_params)
    Inspect.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/content:inspect', http_method='POST', method_id='dlp.projects.content.inspect', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/content:inspect', request_field='googlePrivacyDlpV2InspectContentRequest', request_type_name='DlpProjectsContentInspectRequest', response_type_name='GooglePrivacyDlpV2InspectContentResponse', supports_download=False)

    def Reidentify(self, request, global_params=None):
        """Re-identifies content that has been de-identified. See https://cloud.google.com/sensitive-data-protection/docs/pseudonymization#re-identification_in_free_text_code_example to learn more.

      Args:
        request: (DlpProjectsContentReidentifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ReidentifyContentResponse) The response message.
      """
        config = self.GetMethodConfig('Reidentify')
        return self._RunMethod(config, request, global_params=global_params)
    Reidentify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/content:reidentify', http_method='POST', method_id='dlp.projects.content.reidentify', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/content:reidentify', request_field='googlePrivacyDlpV2ReidentifyContentRequest', request_type_name='DlpProjectsContentReidentifyRequest', response_type_name='GooglePrivacyDlpV2ReidentifyContentResponse', supports_download=False)