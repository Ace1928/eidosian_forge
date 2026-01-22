from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.configdelivery.v1alpha import configdelivery_v1alpha_messages as messages
class ProjectsLocationsFleetPackagesService(base_api.BaseApiService):
    """Service class for the projects_locations_fleetPackages resource."""
    _NAME = 'projects_locations_fleetPackages'

    def __init__(self, client):
        super(ConfigdeliveryV1alpha.ProjectsLocationsFleetPackagesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FleetPackage in a given project and location.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages', http_method='POST', method_id='configdelivery.projects.locations.fleetPackages.create', ordered_params=['parent'], path_params=['parent'], query_params=['fleetPackageId', 'requestId'], relative_path='v1alpha/{+parent}/fleetPackages', request_field='fleetPackage', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single FleetPackage.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}', http_method='DELETE', method_id='configdelivery.projects.locations.fleetPackages.delete', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'force', 'requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single FleetPackage.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FleetPackage) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}', http_method='GET', method_id='configdelivery.projects.locations.fleetPackages.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesGetRequest', response_type_name='FleetPackage', supports_download=False)

    def List(self, request, global_params=None):
        """Lists FleetPackages in a given project and location.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFleetPackagesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages', http_method='GET', method_id='configdelivery.projects.locations.fleetPackages.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/fleetPackages', request_field='', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesListRequest', response_type_name='ListFleetPackagesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single FleetPackage.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/fleetPackages/{fleetPackagesId}', http_method='PATCH', method_id='configdelivery.projects.locations.fleetPackages.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='fleetPackage', request_type_name='ConfigdeliveryProjectsLocationsFleetPackagesPatchRequest', response_type_name='Operation', supports_download=False)