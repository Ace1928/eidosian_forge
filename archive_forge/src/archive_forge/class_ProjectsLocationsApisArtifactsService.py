from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigeeregistry.v1 import apigeeregistry_v1_messages as messages
class ProjectsLocationsApisArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_apis_artifacts resource."""
    _NAME = 'projects_locations_apis_artifacts'

    def __init__(self, client):
        super(ApigeeregistryV1.ProjectsLocationsApisArtifactsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a specified artifact.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts', http_method='POST', method_id='apigeeregistry.projects.locations.apis.artifacts.create', ordered_params=['parent'], path_params=['parent'], query_params=['artifactId'], relative_path='v1/{+parent}/artifacts', request_field='artifact', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsCreateRequest', response_type_name='Artifact', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a specified artifact.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}', http_method='DELETE', method_id='apigeeregistry.projects.locations.apis.artifacts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a specified artifact.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}', http_method='GET', method_id='apigeeregistry.projects.locations.apis.artifacts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsGetRequest', response_type_name='Artifact', supports_download=False)

    def GetContents(self, request, global_params=None):
        """Returns the contents of a specified artifact. If artifacts are stored with GZip compression, the default behavior is to return the artifact uncompressed (the mime_type response field indicates the exact format returned).

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsGetContentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('GetContents')
        return self._RunMethod(config, request, global_params=global_params)
    GetContents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}:getContents', http_method='GET', method_id='apigeeregistry.projects.locations.apis.artifacts.getContents', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:getContents', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsGetContentsRequest', response_type_name='HttpBody', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}:getIamPolicy', http_method='GET', method_id='apigeeregistry.projects.locations.apis.artifacts.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns matching artifacts.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListArtifactsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts', http_method='GET', method_id='apigeeregistry.projects.locations.apis.artifacts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/artifacts', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsListRequest', response_type_name='ListArtifactsResponse', supports_download=False)

    def ReplaceArtifact(self, request, global_params=None):
        """Used to replace a specified artifact.

      Args:
        request: (Artifact) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
        config = self.GetMethodConfig('ReplaceArtifact')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceArtifact.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}', http_method='PUT', method_id='apigeeregistry.projects.locations.apis.artifacts.replaceArtifact', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='Artifact', response_type_name='Artifact', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}:setIamPolicy', http_method='POST', method_id='apigeeregistry.projects.locations.apis.artifacts.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ApigeeregistryProjectsLocationsApisArtifactsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/artifacts/{artifactsId}:testIamPermissions', http_method='POST', method_id='apigeeregistry.projects.locations.apis.artifacts.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ApigeeregistryProjectsLocationsApisArtifactsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)