from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
class ProjectsLocationsGlobalPeeringsService(base_api.BaseApiService):
    """Service class for the projects_locations_global_peerings resource."""
    _NAME = 'projects_locations_global_peerings'

    def __init__(self, client):
        super(ManagedidentitiesV1.ProjectsLocationsGlobalPeeringsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Peering for Managed AD instance.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings', http_method='POST', method_id='managedidentities.projects.locations.global.peerings.create', ordered_params=['parent'], path_params=['parent'], query_params=['peeringId'], relative_path='v1/{+parent}/peerings', request_field='peering', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes identified Peering.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings/{peeringsId}', http_method='DELETE', method_id='managedidentities.projects.locations.global.peerings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Peering.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Peering) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings/{peeringsId}', http_method='GET', method_id='managedidentities.projects.locations.global.peerings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsGetRequest', response_type_name='Peering', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings/{peeringsId}:getIamPolicy', http_method='GET', method_id='managedidentities.projects.locations.global.peerings.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Peerings in a given project.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPeeringsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings', http_method='GET', method_id='managedidentities.projects.locations.global.peerings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/peerings', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsListRequest', response_type_name='ListPeeringsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the labels for specified Peering.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings/{peeringsId}', http_method='PATCH', method_id='managedidentities.projects.locations.global.peerings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='peering', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings/{peeringsId}:setIamPolicy', http_method='POST', method_id='managedidentities.projects.locations.global.peerings.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalPeeringsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/peerings/{peeringsId}:testIamPermissions', http_method='POST', method_id='managedidentities.projects.locations.global.peerings.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ManagedidentitiesProjectsLocationsGlobalPeeringsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)