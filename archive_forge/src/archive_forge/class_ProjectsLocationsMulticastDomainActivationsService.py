from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsMulticastDomainActivationsService(base_api.BaseApiService):
    """Service class for the projects_locations_multicastDomainActivations resource."""
    _NAME = 'projects_locations_multicastDomainActivations'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsMulticastDomainActivationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new multicast domain activation in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainActivationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomainActivations', http_method='POST', method_id='networkservices.projects.locations.multicastDomainActivations.create', ordered_params=['parent'], path_params=['parent'], query_params=['multicastDomainActivationId', 'requestId'], relative_path='v1/{+parent}/multicastDomainActivations', request_field='multicastDomainActivation', request_type_name='NetworkservicesProjectsLocationsMulticastDomainActivationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single multicast domain activation.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainActivationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomainActivations/{multicastDomainActivationsId}', http_method='DELETE', method_id='networkservices.projects.locations.multicastDomainActivations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastDomainActivationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single multicast domain activation.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainActivationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MulticastDomainActivation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomainActivations/{multicastDomainActivationsId}', http_method='GET', method_id='networkservices.projects.locations.multicastDomainActivations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastDomainActivationsGetRequest', response_type_name='MulticastDomainActivation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists multicast domain activations in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainActivationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMulticastDomainActivationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomainActivations', http_method='GET', method_id='networkservices.projects.locations.multicastDomainActivations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/multicastDomainActivations', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastDomainActivationsListRequest', response_type_name='ListMulticastDomainActivationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single multicast domain activation.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainActivationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomainActivations/{multicastDomainActivationsId}', http_method='PATCH', method_id='networkservices.projects.locations.multicastDomainActivations.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='multicastDomainActivation', request_type_name='NetworkservicesProjectsLocationsMulticastDomainActivationsPatchRequest', response_type_name='Operation', supports_download=False)