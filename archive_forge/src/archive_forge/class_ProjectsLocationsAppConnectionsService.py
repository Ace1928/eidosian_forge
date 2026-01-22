from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class ProjectsLocationsAppConnectionsService(base_api.BaseApiService):
    """Service class for the projects_locations_appConnections resource."""
    _NAME = 'projects_locations_appConnections'

    def __init__(self, client):
        super(BeyondcorpV1alpha.ProjectsLocationsAppConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new AppConnection in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections', http_method='POST', method_id='beyondcorp.projects.locations.appConnections.create', ordered_params=['parent'], path_params=['parent'], query_params=['appConnectionId', 'requestId', 'validateOnly'], relative_path='v1alpha/{+parent}/appConnections', request_field='googleCloudBeyondcorpAppconnectionsV1alphaAppConnection', request_type_name='BeyondcorpProjectsLocationsAppConnectionsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single AppConnection.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections/{appConnectionsId}', http_method='DELETE', method_id='beyondcorp.projects.locations.appConnections.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single AppConnection.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpAppconnectionsV1alphaAppConnection) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections/{appConnectionsId}', http_method='GET', method_id='beyondcorp.projects.locations.appConnections.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectionsGetRequest', response_type_name='GoogleCloudBeyondcorpAppconnectionsV1alphaAppConnection', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections/{appConnectionsId}:getIamPolicy', http_method='GET', method_id='beyondcorp.projects.locations.appConnections.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectionsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists AppConnections in a given project and location.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpAppconnectionsV1alphaListAppConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections', http_method='GET', method_id='beyondcorp.projects.locations.appConnections.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/appConnections', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectionsListRequest', response_type_name='GoogleCloudBeyondcorpAppconnectionsV1alphaListAppConnectionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single AppConnection.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections/{appConnectionsId}', http_method='PATCH', method_id='beyondcorp.projects.locations.appConnections.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'requestId', 'updateMask', 'validateOnly'], relative_path='v1alpha/{+name}', request_field='googleCloudBeyondcorpAppconnectionsV1alphaAppConnection', request_type_name='BeyondcorpProjectsLocationsAppConnectionsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Resolve(self, request, global_params=None):
        """Resolves AppConnections details for a given AppConnector. An internal method called by a connector to find AppConnections to connect to.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsResolveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpAppconnectionsV1alphaResolveAppConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('Resolve')
        return self._RunMethod(config, request, global_params=global_params)
    Resolve.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections:resolve', http_method='GET', method_id='beyondcorp.projects.locations.appConnections.resolve', ordered_params=['parent'], path_params=['parent'], query_params=['appConnectorId', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/appConnections:resolve', request_field='', request_type_name='BeyondcorpProjectsLocationsAppConnectionsResolveRequest', response_type_name='GoogleCloudBeyondcorpAppconnectionsV1alphaResolveAppConnectionsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections/{appConnectionsId}:setIamPolicy', http_method='POST', method_id='beyondcorp.projects.locations.appConnections.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='BeyondcorpProjectsLocationsAppConnectionsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BeyondcorpProjectsLocationsAppConnectionsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/appConnections/{appConnectionsId}:testIamPermissions', http_method='POST', method_id='beyondcorp.projects.locations.appConnections.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='BeyondcorpProjectsLocationsAppConnectionsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)