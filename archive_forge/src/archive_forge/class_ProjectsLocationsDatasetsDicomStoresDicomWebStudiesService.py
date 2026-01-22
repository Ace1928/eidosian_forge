from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
class ProjectsLocationsDatasetsDicomStoresDicomWebStudiesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dicomStores_dicomWeb_studies resource."""
    _NAME = 'projects_locations_datasets_dicomStores_dicomWeb_studies'

    def __init__(self, client):
        super(HealthcareV1.ProjectsLocationsDatasetsDicomStoresDicomWebStudiesService, self).__init__(client)
        self._upload_configs = {}

    def GetStudyMetrics(self, request, global_params=None):
        """GetStudyMetrics returns metrics for a study.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesGetStudyMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StudyMetrics) The response message.
      """
        config = self.GetMethodConfig('GetStudyMetrics')
        return self._RunMethod(config, request, global_params=global_params)
    GetStudyMetrics.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dicomStores/{dicomStoresId}/dicomWeb/studies/{studiesId}:getStudyMetrics', http_method='GET', method_id='healthcare.projects.locations.datasets.dicomStores.dicomWeb.studies.getStudyMetrics', ordered_params=['study'], path_params=['study'], query_params=[], relative_path='v1/{+study}:getStudyMetrics', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesGetStudyMetricsRequest', response_type_name='StudyMetrics', supports_download=False)