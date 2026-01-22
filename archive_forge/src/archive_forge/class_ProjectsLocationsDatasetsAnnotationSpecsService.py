from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsDatasetsAnnotationSpecsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_annotationSpecs resource."""
    _NAME = 'projects_locations_datasets_annotationSpecs'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsDatasetsAnnotationSpecsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets an AnnotationSpec.

      Args:
        request: (AiplatformProjectsLocationsDatasetsAnnotationSpecsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AnnotationSpec) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/annotationSpecs/{annotationSpecsId}', http_method='GET', method_id='aiplatform.projects.locations.datasets.annotationSpecs.get', ordered_params=['name'], path_params=['name'], query_params=['readMask'], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsDatasetsAnnotationSpecsGetRequest', response_type_name='GoogleCloudAiplatformV1AnnotationSpec', supports_download=False)