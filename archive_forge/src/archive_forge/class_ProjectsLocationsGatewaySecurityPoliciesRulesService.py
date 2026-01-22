from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsGatewaySecurityPoliciesRulesService(base_api.BaseApiService):
    """Service class for the projects_locations_gatewaySecurityPolicies_rules resource."""
    _NAME = 'projects_locations_gatewaySecurityPolicies_rules'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsGatewaySecurityPoliciesRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new GatewaySecurityPolicy in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}/rules', http_method='POST', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.rules.create', ordered_params=['parent'], path_params=['parent'], query_params=['gatewaySecurityPolicyRuleId'], relative_path='v1/{+parent}/rules', request_field='gatewaySecurityPolicyRule', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single GatewaySecurityPolicyRule.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}/rules/{rulesId}', http_method='DELETE', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.rules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single GatewaySecurityPolicyRule.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GatewaySecurityPolicyRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}/rules/{rulesId}', http_method='GET', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.rules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesGetRequest', response_type_name='GatewaySecurityPolicyRule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists GatewaySecurityPolicyRules in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGatewaySecurityPolicyRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}/rules', http_method='GET', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.rules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/rules', request_field='', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesListRequest', response_type_name='ListGatewaySecurityPolicyRulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single GatewaySecurityPolicyRule.

      Args:
        request: (NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/gatewaySecurityPolicies/{gatewaySecurityPoliciesId}/rules/{rulesId}', http_method='PATCH', method_id='networksecurity.projects.locations.gatewaySecurityPolicies.rules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='gatewaySecurityPolicyRule', request_type_name='NetworksecurityProjectsLocationsGatewaySecurityPoliciesRulesPatchRequest', response_type_name='Operation', supports_download=False)