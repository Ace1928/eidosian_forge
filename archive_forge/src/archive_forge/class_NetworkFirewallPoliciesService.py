from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class NetworkFirewallPoliciesService(base_api.BaseApiService):
    """Service class for the networkFirewallPolicies resource."""
    _NAME = 'networkFirewallPolicies'

    def __init__(self, client):
        super(ComputeBeta.NetworkFirewallPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AddAssociation(self, request, global_params=None):
        """Inserts an association for the specified firewall policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesAddAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    AddAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.addAssociation', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['replaceExistingAssociation', 'requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/addAssociation', request_field='firewallPolicyAssociation', request_type_name='ComputeNetworkFirewallPoliciesAddAssociationRequest', response_type_name='Operation', supports_download=False)

    def AddRule(self, request, global_params=None):
        """Inserts a rule into a firewall policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesAddRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddRule')
        return self._RunMethod(config, request, global_params=global_params)
    AddRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.addRule', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['maxPriority', 'minPriority', 'requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/addRule', request_field='firewallPolicyRule', request_type_name='ComputeNetworkFirewallPoliciesAddRuleRequest', response_type_name='Operation', supports_download=False)

    def CloneRules(self, request, global_params=None):
        """Copies rules to the specified firewall policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesCloneRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CloneRules')
        return self._RunMethod(config, request, global_params=global_params)
    CloneRules.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.cloneRules', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['requestId', 'sourceFirewallPolicy'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/cloneRules', request_field='', request_type_name='ComputeNetworkFirewallPoliciesCloneRulesRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.networkFirewallPolicies.delete', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}', request_field='', request_type_name='ComputeNetworkFirewallPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified network firewall policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkFirewallPolicies.get', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=[], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}', request_field='', request_type_name='ComputeNetworkFirewallPoliciesGetRequest', response_type_name='FirewallPolicy', supports_download=False)

    def GetAssociation(self, request, global_params=None):
        """Gets an association with the specified name.

      Args:
        request: (ComputeNetworkFirewallPoliciesGetAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyAssociation) The response message.
      """
        config = self.GetMethodConfig('GetAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    GetAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkFirewallPolicies.getAssociation', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['name'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/getAssociation', request_field='', request_type_name='ComputeNetworkFirewallPoliciesGetAssociationRequest', response_type_name='FirewallPolicyAssociation', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. May be empty if no such policy or resource exists.

      Args:
        request: (ComputeNetworkFirewallPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkFirewallPolicies.getIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=['optionsRequestedPolicyVersion'], relative_path='projects/{project}/global/firewallPolicies/{resource}/getIamPolicy', request_field='', request_type_name='ComputeNetworkFirewallPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetRule(self, request, global_params=None):
        """Gets a rule of the specified priority.

      Args:
        request: (ComputeNetworkFirewallPoliciesGetRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyRule) The response message.
      """
        config = self.GetMethodConfig('GetRule')
        return self._RunMethod(config, request, global_params=global_params)
    GetRule.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkFirewallPolicies.getRule', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['priority'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/getRule', request_field='', request_type_name='ComputeNetworkFirewallPoliciesGetRuleRequest', response_type_name='FirewallPolicyRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new policy in the specified project using the data included in the request.

      Args:
        request: (ComputeNetworkFirewallPoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/firewallPolicies', request_field='firewallPolicy', request_type_name='ComputeNetworkFirewallPoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the policies that have been configured for the specified project.

      Args:
        request: (ComputeNetworkFirewallPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallPolicyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.networkFirewallPolicies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/firewallPolicies', request_field='', request_type_name='ComputeNetworkFirewallPoliciesListRequest', response_type_name='FirewallPolicyList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified policy with the data included in the request.

      Args:
        request: (ComputeNetworkFirewallPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.networkFirewallPolicies.patch', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}', request_field='firewallPolicyResource', request_type_name='ComputeNetworkFirewallPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchRule(self, request, global_params=None):
        """Patches a rule of the specified priority.

      Args:
        request: (ComputeNetworkFirewallPoliciesPatchRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchRule')
        return self._RunMethod(config, request, global_params=global_params)
    PatchRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.patchRule', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['priority', 'requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/patchRule', request_field='firewallPolicyRule', request_type_name='ComputeNetworkFirewallPoliciesPatchRuleRequest', response_type_name='Operation', supports_download=False)

    def RemoveAssociation(self, request, global_params=None):
        """Removes an association for the specified firewall policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesRemoveAssociationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveAssociation')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveAssociation.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.removeAssociation', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['name', 'requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/removeAssociation', request_field='', request_type_name='ComputeNetworkFirewallPoliciesRemoveAssociationRequest', response_type_name='Operation', supports_download=False)

    def RemoveRule(self, request, global_params=None):
        """Deletes a rule of the specified priority.

      Args:
        request: (ComputeNetworkFirewallPoliciesRemoveRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveRule')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.removeRule', ordered_params=['project', 'firewallPolicy'], path_params=['firewallPolicy', 'project'], query_params=['priority', 'requestId'], relative_path='projects/{project}/global/firewallPolicies/{firewallPolicy}/removeRule', request_field='', request_type_name='ComputeNetworkFirewallPoliciesRemoveRuleRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.

      Args:
        request: (ComputeNetworkFirewallPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.setIamPolicy', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/firewallPolicies/{resource}/setIamPolicy', request_field='globalSetPolicyRequest', request_type_name='ComputeNetworkFirewallPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeNetworkFirewallPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.networkFirewallPolicies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/firewallPolicies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeNetworkFirewallPoliciesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)