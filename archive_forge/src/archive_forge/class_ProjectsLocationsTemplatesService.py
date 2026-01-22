from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
class ProjectsLocationsTemplatesService(base_api.BaseApiService):
    """Service class for the projects_locations_templates resource."""
    _NAME = 'projects_locations_templates'

    def __init__(self, client):
        super(DataflowV1b3.ProjectsLocationsTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Cloud Dataflow job from a template. Do not enter confidential information when you supply string values using the API.

      Args:
        request: (DataflowProjectsLocationsTemplatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Job) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.locations.templates.create', ordered_params=['projectId', 'location'], path_params=['location', 'projectId'], query_params=[], relative_path='v1b3/projects/{projectId}/locations/{location}/templates', request_field='createJobFromTemplateRequest', request_type_name='DataflowProjectsLocationsTemplatesCreateRequest', response_type_name='Job', supports_download=False)

    def Get(self, request, global_params=None):
        """Get the template associated with a template.

      Args:
        request: (DataflowProjectsLocationsTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetTemplateResponse) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dataflow.projects.locations.templates.get', ordered_params=['projectId', 'location'], path_params=['location', 'projectId'], query_params=['gcsPath', 'view'], relative_path='v1b3/projects/{projectId}/locations/{location}/templates:get', request_field='', request_type_name='DataflowProjectsLocationsTemplatesGetRequest', response_type_name='GetTemplateResponse', supports_download=False)

    def Launch(self, request, global_params=None):
        """Launch a template.

      Args:
        request: (DataflowProjectsLocationsTemplatesLaunchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LaunchTemplateResponse) The response message.
      """
        config = self.GetMethodConfig('Launch')
        return self._RunMethod(config, request, global_params=global_params)
    Launch.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dataflow.projects.locations.templates.launch', ordered_params=['projectId', 'location'], path_params=['location', 'projectId'], query_params=['dynamicTemplate_gcsPath', 'dynamicTemplate_stagingLocation', 'gcsPath', 'validateOnly'], relative_path='v1b3/projects/{projectId}/locations/{location}/templates:launch', request_field='launchTemplateParameters', request_type_name='DataflowProjectsLocationsTemplatesLaunchRequest', response_type_name='LaunchTemplateResponse', supports_download=False)