from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsAssetTypesAssetsAnnotationSetsService(base_api.BaseApiService):
    """Service class for the projects_locations_assetTypes_assets_annotationSets resource."""
    _NAME = 'projects_locations_assetTypes_assets_annotationSets'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsAssetTypesAssetsAnnotationSetsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single annotationSet.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnnotationSet) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsGetRequest', response_type_name='AnnotationSet', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}:getIamPolicy', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists annotationSets in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAnnotationSetsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/annotationSets', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsListRequest', response_type_name='ListAnnotationSetsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}:setIamPolicy', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/assets/{assetsId}/annotationSets/{annotationSetsId}:testIamPermissions', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.assets.annotationSets.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='MediaassetProjectsLocationsAssetTypesAssetsAnnotationSetsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)