from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsPrivateCloudsExternalAddressesService(base_api.BaseApiService):
    """Service class for the projects_locations_privateClouds_externalAddresses resource."""
    _NAME = 'projects_locations_privateClouds_externalAddresses'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsPrivateCloudsExternalAddressesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `ExternalAddress` resource in a given private cloud. The network policy that corresponds to the private cloud must have the external IP address network service enabled (`NetworkPolicy.external_ip`).

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsExternalAddressesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/externalAddresses', http_method='POST', method_id='vmwareengine.projects.locations.privateClouds.externalAddresses.create', ordered_params=['parent'], path_params=['parent'], query_params=['externalAddressId', 'requestId'], relative_path='v1/{+parent}/externalAddresses', request_field='externalAddress', request_type_name='VmwareengineProjectsLocationsPrivateCloudsExternalAddressesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single external IP address. When you delete an external IP address, connectivity between the external IP address and the corresponding internal IP address is lost.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsExternalAddressesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/externalAddresses/{externalAddressesId}', http_method='DELETE', method_id='vmwareengine.projects.locations.privateClouds.externalAddresses.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsExternalAddressesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single external IP address.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsExternalAddressesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExternalAddress) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/externalAddresses/{externalAddressesId}', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.externalAddresses.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsExternalAddressesGetRequest', response_type_name='ExternalAddress', supports_download=False)

    def List(self, request, global_params=None):
        """Lists external IP addresses assigned to VMware workload VMs in a given private cloud.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsExternalAddressesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExternalAddressesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/externalAddresses', http_method='GET', method_id='vmwareengine.projects.locations.privateClouds.externalAddresses.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/externalAddresses', request_field='', request_type_name='VmwareengineProjectsLocationsPrivateCloudsExternalAddressesListRequest', response_type_name='ListExternalAddressesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single external IP address. Only fields specified in `update_mask` are applied. During operation processing, the resource is temporarily in the `ACTIVE` state before the operation fully completes. For that period of time, you can't update the resource. Use the operation status to determine when the processing fully completes.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsExternalAddressesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/privateClouds/{privateCloudsId}/externalAddresses/{externalAddressesId}', http_method='PATCH', method_id='vmwareengine.projects.locations.privateClouds.externalAddresses.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='externalAddress', request_type_name='VmwareengineProjectsLocationsPrivateCloudsExternalAddressesPatchRequest', response_type_name='Operation', supports_download=False)