from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsNetworkPoliciesExternalAccessRulesService(base_api.BaseApiService):
    """Service class for the projects_locations_networkPolicies_externalAccessRules resource."""
    _NAME = 'projects_locations_networkPolicies_externalAccessRules'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsNetworkPoliciesExternalAccessRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new external access rule in a given network policy.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}/externalAccessRules', http_method='POST', method_id='vmwareengine.projects.locations.networkPolicies.externalAccessRules.create', ordered_params=['parent'], path_params=['parent'], query_params=['externalAccessRuleId', 'requestId'], relative_path='v1/{+parent}/externalAccessRules', request_field='externalAccessRule', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single external access rule.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}/externalAccessRules/{externalAccessRulesId}', http_method='DELETE', method_id='vmwareengine.projects.locations.networkPolicies.externalAccessRules.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single external access rule.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExternalAccessRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}/externalAccessRules/{externalAccessRulesId}', http_method='GET', method_id='vmwareengine.projects.locations.networkPolicies.externalAccessRules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesGetRequest', response_type_name='ExternalAccessRule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `ExternalAccessRule` resources in the specified network policy.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListExternalAccessRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}/externalAccessRules', http_method='GET', method_id='vmwareengine.projects.locations.networkPolicies.externalAccessRules.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/externalAccessRules', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesListRequest', response_type_name='ListExternalAccessRulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single external access rule. Only fields specified in `update_mask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPolicies/{networkPoliciesId}/externalAccessRules/{externalAccessRulesId}', http_method='PATCH', method_id='vmwareengine.projects.locations.networkPolicies.externalAccessRules.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='externalAccessRule', request_type_name='VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesPatchRequest', response_type_name='Operation', supports_download=False)