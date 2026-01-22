from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.transferappliance.v1alpha1 import transferappliance_v1alpha1_messages as messages
class ProjectsLocationsSavedAddressesService(base_api.BaseApiService):
    """Service class for the projects_locations_savedAddresses resource."""
    _NAME = 'projects_locations_savedAddresses'

    def __init__(self, client):
        super(TransferapplianceV1alpha1.ProjectsLocationsSavedAddressesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new SavedAddress in a given project and location.

      Args:
        request: (TransferapplianceProjectsLocationsSavedAddressesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/savedAddresses', http_method='POST', method_id='transferappliance.projects.locations.savedAddresses.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'savedAddressId', 'validateOnly'], relative_path='v1alpha1/{+parent}/savedAddresses', request_field='savedAddress', request_type_name='TransferapplianceProjectsLocationsSavedAddressesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single SavedAddress.

      Args:
        request: (TransferapplianceProjectsLocationsSavedAddressesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/savedAddresses/{savedAddressesId}', http_method='DELETE', method_id='transferappliance.projects.locations.savedAddresses.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId', 'validateOnly'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='TransferapplianceProjectsLocationsSavedAddressesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single SavedAddress.

      Args:
        request: (TransferapplianceProjectsLocationsSavedAddressesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SavedAddress) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/savedAddresses/{savedAddressesId}', http_method='GET', method_id='transferappliance.projects.locations.savedAddresses.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='TransferapplianceProjectsLocationsSavedAddressesGetRequest', response_type_name='SavedAddress', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SavedAddresses in a given project and location.

      Args:
        request: (TransferapplianceProjectsLocationsSavedAddressesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSavedAddressesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/savedAddresses', http_method='GET', method_id='transferappliance.projects.locations.savedAddresses.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/savedAddresses', request_field='', request_type_name='TransferapplianceProjectsLocationsSavedAddressesListRequest', response_type_name='ListSavedAddressesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single SavedAddress.

      Args:
        request: (TransferapplianceProjectsLocationsSavedAddressesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/savedAddresses/{savedAddressesId}', http_method='PATCH', method_id='transferappliance.projects.locations.savedAddresses.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1alpha1/{+name}', request_field='savedAddress', request_type_name='TransferapplianceProjectsLocationsSavedAddressesPatchRequest', response_type_name='Operation', supports_download=False)