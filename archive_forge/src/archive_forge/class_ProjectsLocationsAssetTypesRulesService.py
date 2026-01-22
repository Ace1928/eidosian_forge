from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsAssetTypesRulesService(base_api.BaseApiService):
    """Service class for the projects_locations_assetTypes_rules resource."""
    _NAME = 'projects_locations_assetTypes_rules'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsAssetTypesRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new rule in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/rules', http_method='POST', method_id='mediaasset.projects.locations.assetTypes.rules.create', ordered_params=['parent'], path_params=['parent'], query_params=['ruleId'], relative_path='v1alpha/{+parent}/rules', request_field='rule', request_type_name='MediaassetProjectsLocationsAssetTypesRulesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single rule.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/rules/{rulesId}', http_method='DELETE', method_id='mediaasset.projects.locations.assetTypes.rules.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesRulesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single rule.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Rule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/rules/{rulesId}', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.rules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesRulesGetRequest', response_type_name='Rule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists rules in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/rules', http_method='GET', method_id='mediaasset.projects.locations.assetTypes.rules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/rules', request_field='', request_type_name='MediaassetProjectsLocationsAssetTypesRulesListRequest', response_type_name='ListRulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single rule.

      Args:
        request: (MediaassetProjectsLocationsAssetTypesRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/assetTypes/{assetTypesId}/rules/{rulesId}', http_method='PATCH', method_id='mediaasset.projects.locations.assetTypes.rules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='rule', request_type_name='MediaassetProjectsLocationsAssetTypesRulesPatchRequest', response_type_name='Operation', supports_download=False)