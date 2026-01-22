from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsAssetTypesAssetsService(base_api.BaseApiService):
    """Service class for the projects_locations_assetTypes_assets resource."""
    _NAME = 'projects_locations_assetTypes_assets'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsAssetTypesAssetsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new asset in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.create', ordered_params=['parent'], path_params=['parent'], query_params=['assetId'], relative_path='v1alpha/{+parent}/assets', request_field='asset', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single asset.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}', http_method='DELETE', method_id='mediaasset.projects.locations.assetTypes.assets.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single asset.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Asset) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.get', ordered_params=['name'], path_params=['name'], query_params=['readMask'], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsGetRequest', response_type_name='Asset', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}:getIamPolicy', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists assets in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAssetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken', 'readMask'], relative_path='v1alpha/{+parent}/assets', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsListRequest', response_type_name='ListAssetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single asset.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}', http_method='PATCH', method_id='mediaasset.projects.locations.assetTypes.assets.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='asset', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}:setIamPolicy', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}:testIamPermissions', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)