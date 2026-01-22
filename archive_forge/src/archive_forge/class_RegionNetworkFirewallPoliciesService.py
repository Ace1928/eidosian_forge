from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class RegionNetworkFirewallPoliciesService(base_api.BaseApiService):
    """Service class for the regionNetworkFirewallPolicies resource."""
    _NAME = 'regionNetworkFirewallPolicies'

    def __init__(self, client):
        super(ComputeBeta.RegionNetworkFirewallPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AddAssociation(self, request, global_params=None):
        """Inserts an association for the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesAddAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    AddAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.addAssociation', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['replaceExistingAssociation', 'requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/addAssociation', request_field='firewallPolicyAssociation', request_type_name='ComputeRegionNetworkFirewallPoliciesAddAssociationRequest', response_type_name='Operation', supports_download=False)

    def AddRule(self, request, global_params=None):
        """Inserts a rule into a network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesAddRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddRule')
        return self._RunMethod(config, request, global_params=global_params)
    AddRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.addRule', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['maxPriority', 'minPriority', 'requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/addRule', request_field='firewallPolicyRule', request_type_name='ComputeRegionNetworkFirewallPoliciesAddRuleRequest', response_type_name='Operation', supports_download=False)

    def CloneRules(self, request, global_params=None):
        """Copies rules to the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesCloneRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CloneRules')
        return self._RunMethod(config, request, global_params=global_params)
    CloneRules.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.cloneRules', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['requestId', 'sourceFirewallPolicy'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/cloneRules', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesCloneRulesRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.regionNetworkFirewallPolicies.delete', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkFirewallPolicies.get', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=[], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesGetRequest', response_type_name='FirewallPolicy', supports_download=False)

    def GetAssociation(self, request, global_params=None):
        """Gets an association with the specified name.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesGetAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyAssociation) The response message.
      """
        config = self.GetMethodConfig('GetAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    GetAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkFirewallPolicies.getAssociation', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['name'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/getAssociation', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesGetAssociationRequest', response_type_name='FirewallPolicyAssociation', supports_download=False)

    def GetEffectiveFirewalls(self, request, global_params=None):
        """Returns the effective firewalls on a given network.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesGetEffectiveFirewallsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponse) The response message.
      """
        config = self.GetMethodConfig('GetEffectiveFirewalls')
        return self._RunMethod(config, request, global_params=global_params)
    GetEffectiveFirewalls.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkFirewallPolicies.getEffectiveFirewalls', ordered_params=['project', 'region', 'network'], path_params=['project', 'region'], query_params=['network'], relative_path='projects/{project}/regions/{region}/firewallPolicies/getEffectiveFirewalls', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesGetEffectiveFirewallsRequest', response_type_name='RegionNetworkFirewallPoliciesGetEffectiveFirewallsResponse', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkFirewallPolicies.getIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{resource}/getIamPolicy', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetRule(self, request, global_params=None):
        """Gets a rule of the specified priority.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesGetRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyRule) The response message.
      """
        config = self.GetMethodConfig('GetRule')
        return self._RunMethod(config, request, global_params=global_params)
    GetRule.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkFirewallPolicies.getRule', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['priority'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/getRule', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesGetRuleRequest', response_type_name='FirewallPolicyRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new network firewall policy in the specified project and region.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.insert', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies', request_field='firewallPolicy', request_type_name='ComputeRegionNetworkFirewallPoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the network firewall policies that have been configured for the specified project in the given region.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.regionNetworkFirewallPolicies.list', ordered_params=['project', 'region'], path_params=['project', 'region'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/regions/{region}/firewallPolicies', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesListRequest', response_type_name='FirewallPolicyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.regionNetworkFirewallPolicies.patch', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}', request_field='firewallPolicyResource', request_type_name='ComputeRegionNetworkFirewallPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchRule(self, request, global_params=None):
        """Patches a rule of the specified priority.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesPatchRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchRule')
        return self._RunMethod(config, request, global_params=global_params)
    PatchRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.patchRule', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['priority', 'requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/patchRule', request_field='firewallPolicyRule', request_type_name='ComputeRegionNetworkFirewallPoliciesPatchRuleRequest', response_type_name='Operation', supports_download=False)

    def RemoveAssociation(self, request, global_params=None):
        """Removes an association for the specified network firewall policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesRemoveAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.removeAssociation', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['name', 'requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/removeAssociation', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesRemoveAssociationRequest', response_type_name='Operation', supports_download=False)

    def RemoveRule(self, request, global_params=None):
        """Deletes a rule of the specified priority.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesRemoveRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveRule')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.removeRule', ordered_params=['project', 'region', 'firewallPolicy'], path_params=['firewallPolicy', 'project', 'region'], query_params=['priority', 'requestId'], relative_path='projects/{project}/regions/{region}/firewallPolicies/{firewallPolicy}/removeRule', request_field='', request_type_name='ComputeRegionNetworkFirewallPoliciesRemoveRuleRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.setIamPolicy', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/firewallPolicies/{resource}/setIamPolicy', request_field='regionSetPolicyRequest', request_type_name='ComputeRegionNetworkFirewallPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeRegionNetworkFirewallPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.regionNetworkFirewallPolicies.testIamPermissions', ordered_params=['project', 'region', 'resource'], path_params=['project', 'region', 'resource'], query_params=[], relative_path='projects/{project}/regions/{region}/firewallPolicies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeRegionNetworkFirewallPoliciesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)