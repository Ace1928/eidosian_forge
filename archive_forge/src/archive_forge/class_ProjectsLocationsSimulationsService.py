from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkmanagement.v1alpha1 import networkmanagement_v1alpha1_messages as messages
class ProjectsLocationsSimulationsService(base_api.BaseApiService):
    """Service class for the projects_locations_simulations resource."""
    _NAME = 'projects_locations_simulations'

    def __init__(self, client):
        super(NetworkmanagementV1alpha1.ProjectsLocationsSimulationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Simulation in a given project and location. After user creates a simulation, the simulation is performed as part of the long running operation.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations', http_method='POST', method_id='networkmanagement.projects.locations.simulations.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1alpha1/{+parent}/simulations', request_field='simulation', request_type_name='NetworkmanagementProjectsLocationsSimulationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Simulation.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations/{simulationsId}', http_method='DELETE', method_id='networkmanagement.projects.locations.simulations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsSimulationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Simulation.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Simulation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations/{simulationsId}', http_method='GET', method_id='networkmanagement.projects.locations.simulations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsSimulationsGetRequest', response_type_name='Simulation', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations/{simulationsId}:getIamPolicy', http_method='GET', method_id='networkmanagement.projects.locations.simulations.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha1/{+resource}:getIamPolicy', request_field='', request_type_name='NetworkmanagementProjectsLocationsSimulationsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Simulations in a given project and location.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSimulationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations', http_method='GET', method_id='networkmanagement.projects.locations.simulations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/simulations', request_field='', request_type_name='NetworkmanagementProjectsLocationsSimulationsListRequest', response_type_name='ListSimulationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single Simulation.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations/{simulationsId}', http_method='PATCH', method_id='networkmanagement.projects.locations.simulations.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha1/{+name}', request_field='simulation', request_type_name='NetworkmanagementProjectsLocationsSimulationsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations/{simulationsId}:setIamPolicy', http_method='POST', method_id='networkmanagement.projects.locations.simulations.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='NetworkmanagementProjectsLocationsSimulationsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (NetworkmanagementProjectsLocationsSimulationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/simulations/{simulationsId}:testIamPermissions', http_method='POST', method_id='networkmanagement.projects.locations.simulations.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='NetworkmanagementProjectsLocationsSimulationsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)