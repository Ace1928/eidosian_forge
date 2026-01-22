from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsConsentStoresConsentsService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_consentStores_consents resource."""
    _NAME = 'projects_locations_datasets_consentStores_consents'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsConsentStoresConsentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Consent in the parent consent store.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Consent) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consents', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.consents.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/consents', request_field='consent', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentsCreateRequest', response_type_name='Consent', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the Consent and its revisions. To keep a record of the Consent but mark it inactive, see [RevokeConsent]. To delete a revision of a Consent, see [DeleteConsentRevision]. This operation does not delete the related Consent artifact.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consents/{consentsId}', http_method='DELETE', method_id='healthcare.projects.locations.datasets.consentStores.consents.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified revision of a Consent, or the latest revision if `revision_id` is not specified in the resource name.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Consent) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consents/{consentsId}', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.consents.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentsGetRequest', response_type_name='Consent', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Consent in the given consent store, returning each Consent's latest revision.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConsentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consents', http_method='GET', method_id='healthcare.projects.locations.datasets.consentStores.consents.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/consents', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentsListRequest', response_type_name='ListConsentsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the latest revision of the specified Consent by committing a new revision with the changes. A FAILED_PRECONDITION error occurs if the latest revision of the specified Consent is in the `REJECTED` or `REVOKED` state.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Consent) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consents/{consentsId}', http_method='PATCH', method_id='healthcare.projects.locations.datasets.consentStores.consents.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='consent', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentsPatchRequest', response_type_name='Consent', supports_download=False)

    def Revoke(self, request, global_params=None):
        """Revokes the latest revision of the specified Consent by committing a new revision with `state` updated to `REVOKED`. If the latest revision of the specified Consent is in the `REVOKED` state, no new revision is committed. A FAILED_PRECONDITION error occurs if the latest revision of the given consent is in `DRAFT` or `REJECTED` state.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresConsentsRevokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Consent) The response message.
      """
        config = self.GetMethodConfig('Revoke')
        return self._RunMethod(config, request, global_params=global_params)
    Revoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/consentStores/{consentStoresId}/consents/{consentsId}:revoke', http_method='POST', method_id='healthcare.projects.locations.datasets.consentStores.consents.revoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:revoke', request_field='revokeConsentRequest', request_type_name='HealthcareProjectsLocationsDatasetsConsentStoresConsentsRevokeRequest', response_type_name='Consent', supports_download=False)