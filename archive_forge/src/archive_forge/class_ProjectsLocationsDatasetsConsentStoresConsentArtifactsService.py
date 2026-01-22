from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsConsentStoresConsentArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_consentStores_consentArtifacts resource."""
    _NAME = 'projects_locations_datasets_consentStores_consentArtifacts'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsConsentStoresConsentArtifactsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Consent artifact in the parent consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsentArtifact) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consentArtifacts', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.consentArtifacts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/consentArtifacts', request_field='consentArtifact', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsCreateRequest', response_type_name='ConsentArtifact', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified Consent artifact. Fails if the artifact is referenced by the latest revision of any Consent.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consentArtifacts/{consentArtifactsId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.consentStores.consentArtifacts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified Consent artifact.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsentArtifact) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consentArtifacts/{consentArtifactsId}', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.consentArtifacts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsGetRequest', response_type_name='ConsentArtifact', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Consent artifacts in the specified consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConsentArtifactsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consentArtifacts', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.consentArtifacts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/consentArtifacts', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsListRequest', response_type_name='ListConsentArtifactsResponse', supports_download=False)