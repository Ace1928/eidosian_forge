from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsDatasetsDataMapperWorkspacesService(base_api.BaseApiService):
    """Service class for the projects_locations_datasets_dataMapperWorkspaces resource."""
    _NAME = 'projects_locations_datasets_dataMapperWorkspaces'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsDatasetsDataMapperWorkspacesService, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDataMapperWorkspacesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dataMapperWorkspaces/{dataMapperWorkspacesId}:getIamPolicy', http_method='GET', method_id='healthcare.projects.locations.datasets.dataMapperWorkspaces.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='HealthcareProjectsLocationsDatasetsDataMapperWorkspacesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDataMapperWorkspacesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dataMapperWorkspaces/{dataMapperWorkspacesId}:setIamPolicy', http_method='POST', method_id='healthcare.projects.locations.datasets.dataMapperWorkspaces.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='HealthcareProjectsLocationsDatasetsDataMapperWorkspacesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDataMapperWorkspacesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/datasets/{datasetsId}/dataMapperWorkspaces/{dataMapperWorkspacesId}:testIamPermissions', http_method='POST', method_id='healthcare.projects.locations.datasets.dataMapperWorkspaces.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='HealthcareProjectsLocationsDatasetsDataMapperWorkspacesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)