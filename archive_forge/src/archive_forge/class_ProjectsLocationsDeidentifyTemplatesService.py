from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsLocationsDeidentifyTemplatesService(base_api.BaseApiService):
    """Service class for the projects_locations_deidentifyTemplates resource."""
    _NAME = 'projects_locations_deidentifyTemplates'

    def __init__(self, client):
        super(DlpV2.ProjectsLocationsDeidentifyTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a DeidentifyTemplate for reusing frequently used configuration for de-identifying content, images, and storage. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates-deid to learn more.

      Args:
        request: (DlpProjectsLocationsDeidentifyTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DeidentifyTemplate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/deidentifyTemplates', http_method='POST', method_id='dlp.projects.locations.deidentifyTemplates.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/deidentifyTemplates', request_field='googlePrivacyDlpV2CreateDeidentifyTemplateRequest', request_type_name='DlpProjectsLocationsDeidentifyTemplatesCreateRequest', response_type_name='GooglePrivacyDlpV2DeidentifyTemplate', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a DeidentifyTemplate. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates-deid to learn more.

      Args:
        request: (DlpProjectsLocationsDeidentifyTemplatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/deidentifyTemplates/{deidentifyTemplatesId}', http_method='DELETE', method_id='dlp.projects.locations.deidentifyTemplates.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsDeidentifyTemplatesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a DeidentifyTemplate. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates-deid to learn more.

      Args:
        request: (DlpProjectsLocationsDeidentifyTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DeidentifyTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/deidentifyTemplates/{deidentifyTemplatesId}', http_method='GET', method_id='dlp.projects.locations.deidentifyTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsDeidentifyTemplatesGetRequest', response_type_name='GooglePrivacyDlpV2DeidentifyTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DeidentifyTemplates. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates-deid to learn more.

      Args:
        request: (DlpProjectsLocationsDeidentifyTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListDeidentifyTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/deidentifyTemplates', http_method='GET', method_id='dlp.projects.locations.deidentifyTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['locationId', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/deidentifyTemplates', request_field='', request_type_name='DlpProjectsLocationsDeidentifyTemplatesListRequest', response_type_name='GooglePrivacyDlpV2ListDeidentifyTemplatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the DeidentifyTemplate. See https://cloud.google.com/sensitive-data-protection/docs/creating-templates-deid to learn more.

      Args:
        request: (DlpProjectsLocationsDeidentifyTemplatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2DeidentifyTemplate) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/deidentifyTemplates/{deidentifyTemplatesId}', http_method='PATCH', method_id='dlp.projects.locations.deidentifyTemplates.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googlePrivacyDlpV2UpdateDeidentifyTemplateRequest', request_type_name='DlpProjectsLocationsDeidentifyTemplatesPatchRequest', response_type_name='GooglePrivacyDlpV2DeidentifyTemplate', supports_download=False)