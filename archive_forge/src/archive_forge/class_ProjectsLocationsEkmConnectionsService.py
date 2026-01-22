from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudkms.v1 import cloudkms_v1_messages as messages
class ProjectsLocationsEkmConnectionsService(base_api.BaseApiService):
    """Service class for the projects_locations_ekmConnections resource."""
    _NAME = 'projects_locations_ekmConnections'

    def __init__(self, client):
        super(CloudkmsV1.ProjectsLocationsEkmConnectionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new EkmConnection in a given Project and Location.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EkmConnection) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections', http_method='POST', method_id='cloudkms.projects.locations.ekmConnections.create', ordered_params=['parent'], path_params=['parent'], query_params=['ekmConnectionId'], relative_path='v1/{+parent}/ekmConnections', request_field='ekmConnection', request_type_name='CloudkmsProjectsLocationsEkmConnectionsCreateRequest', response_type_name='EkmConnection', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns metadata for a given EkmConnection.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EkmConnection) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections/{ekmConnectionsId}', http_method='GET', method_id='cloudkms.projects.locations.ekmConnections.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudkmsProjectsLocationsEkmConnectionsGetRequest', response_type_name='EkmConnection', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections/{ekmConnectionsId}:getIamPolicy', http_method='GET', method_id='cloudkms.projects.locations.ekmConnections.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='CloudkmsProjectsLocationsEkmConnectionsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists EkmConnections.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListEkmConnectionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections', http_method='GET', method_id='cloudkms.projects.locations.ekmConnections.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/ekmConnections', request_field='', request_type_name='CloudkmsProjectsLocationsEkmConnectionsListRequest', response_type_name='ListEkmConnectionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an EkmConnection's metadata.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EkmConnection) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections/{ekmConnectionsId}', http_method='PATCH', method_id='cloudkms.projects.locations.ekmConnections.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='ekmConnection', request_type_name='CloudkmsProjectsLocationsEkmConnectionsPatchRequest', response_type_name='EkmConnection', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections/{ekmConnectionsId}:setIamPolicy', http_method='POST', method_id='cloudkms.projects.locations.ekmConnections.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CloudkmsProjectsLocationsEkmConnectionsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections/{ekmConnectionsId}:testIamPermissions', http_method='POST', method_id='cloudkms.projects.locations.ekmConnections.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CloudkmsProjectsLocationsEkmConnectionsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def VerifyConnectivity(self, request, global_params=None):
        """Verifies that Cloud KMS can successfully connect to the external key manager specified by an EkmConnection. If there is an error connecting to the EKM, this method returns a FAILED_PRECONDITION status containing structured information as described at https://cloud.google.com/kms/docs/reference/ekm_errors.

      Args:
        request: (CloudkmsProjectsLocationsEkmConnectionsVerifyConnectivityRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VerifyConnectivityResponse) The response message.
      """
        config = self.GetMethodConfig('VerifyConnectivity')
        return self._RunMethod(config, request, global_params=global_params)
    VerifyConnectivity.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/ekmConnections/{ekmConnectionsId}:verifyConnectivity', http_method='GET', method_id='cloudkms.projects.locations.ekmConnections.verifyConnectivity', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:verifyConnectivity', request_field='', request_type_name='CloudkmsProjectsLocationsEkmConnectionsVerifyConnectivityRequest', response_type_name='VerifyConnectivityResponse', supports_download=False)