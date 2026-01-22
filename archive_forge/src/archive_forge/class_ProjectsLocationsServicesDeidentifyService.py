from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsServicesDeidentifyService(base_api.BaseApiService):
    """Service class for the projects_locations_services_deidentify resource."""
    _NAME = 'projects_locations_services_deidentify'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsServicesDeidentifyService, self).__init__(client)
        self._upload_configs = {}

    def DeidentifyDicomInstance(self, request, global_params=None):
        """De-identify a single DICOM instance. Uses the ATTRIBUTE_CONFIDENTIALITY_BASIC_PROFILE TagFilterProfile and the REDACT_ALL_TEXT TextRedactionMode.

      Args:
        request: (HealthcareProjectsLocationsServicesDeidentifyDeidentifyDicomInstanceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('DeidentifyDicomInstance')
        return self._RunMethod(config, request, global_params=global_params)
    DeidentifyDicomInstance.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/services/deidentify:deidentifyDicomInstance', http_method='POST', method_id='healthcare.projects.locations.services.deidentify.deidentifyDicomInstance', ordered_params=['name'], path_params=['name'], query_params=['gcsConfigUri'], relative_path='v1alpha2/{+name}:deidentifyDicomInstance', request_field='httpBody', request_type_name='HealthcareProjectsLocationsServicesDeidentifyDeidentifyDicomInstanceRequest', response_type_name='HttpBody', supports_download=False)

    def DeidentifyFhirResource(self, request, global_params=None):
        """De-identify a single FHIR resource.

      Args:
        request: (HealthcareProjectsLocationsServicesDeidentifyDeidentifyFhirResourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('DeidentifyFhirResource')
        return self._RunMethod(config, request, global_params=global_params)
    DeidentifyFhirResource.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/services/deidentify:deidentifyFhirResource', http_method='POST', method_id='healthcare.projects.locations.services.deidentify.deidentifyFhirResource', ordered_params=['name'], path_params=['name'], query_params=['gcsConfigUri', 'version'], relative_path='v1alpha2/{+name}:deidentifyFhirResource', request_field='httpBody', request_type_name='HealthcareProjectsLocationsServicesDeidentifyDeidentifyFhirResourceRequest', response_type_name='HttpBody', supports_download=False)