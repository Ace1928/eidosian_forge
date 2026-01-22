from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsConsentStoresUserDataMappingsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_consentStores_userDataMappings resource."""
    _NAME = 'projects_locations_datasets_consentStores_userDataMappings'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsConsentStoresUserDataMappingsService, self).__init__(client)
        self._upload_configs = {}

    def Archive(self, request, global_params=None):
        """Archives the specified User data mapping.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsArchiveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ArchiveUserDataMappingResponse) The response message.
      """
        config = self.GetMethodConfig('Archive')
        return self._RunMethod(config, request, global_params=global_params)
    Archive.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/userDataMappings/{userDataMappingsId}:archive', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.userDataMappings.archive', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:archive', request_field='archiveUserDataMappingRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsArchiveRequest', response_type_name='ArchiveUserDataMappingResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new User data mapping in the parent consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserDataMapping) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/userDataMappings', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.userDataMappings.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/userDataMappings', request_field='userDataMapping', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsCreateRequest', response_type_name='UserDataMapping', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified User data mapping.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/userDataMappings/{userDataMappingsId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.consentStores.userDataMappings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified User data mapping.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserDataMapping) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/userDataMappings/{userDataMappingsId}', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.userDataMappings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsGetRequest', response_type_name='UserDataMapping', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the User data mappings in the specified consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUserDataMappingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/userDataMappings', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.userDataMappings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/userDataMappings', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsListRequest', response_type_name='ListUserDataMappingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified User data mapping.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserDataMapping) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/userDataMappings/{userDataMappingsId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.consentStores.userDataMappings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='userDataMapping', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsPatchRequest', response_type_name='UserDataMapping', supports_download=False)