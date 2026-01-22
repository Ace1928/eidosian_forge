from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudiot.v1 import cloudiot_v1_messages as messages
class ProjectsLocationsRegistriesDevicesStatesService(base_api.BaseApiService):
    """Service class for the projects_locations_registries_devices_states resource."""
    _NAME = 'projects_locations_registries_devices_states'

    def __init__(self, client):
        super(CloudiotV1.ProjectsLocationsRegistriesDevicesStatesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the last few versions of the device state in descending order (i.e.: newest first).

      Args:
        request: (CloudiotProjectsLocationsRegistriesDevicesStatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeviceStatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/registries/{registriesId}/devices/{devicesId}/states', http_method='GET', method_id='cloudiot.projects.locations.registries.devices.states.list', ordered_params=['name'], path_params=['name'], query_params=['numStates'], relative_path='v1/{+name}/states', request_field='', request_type_name='CloudiotProjectsLocationsRegistriesDevicesStatesListRequest', response_type_name='ListDeviceStatesResponse', supports_download=False)