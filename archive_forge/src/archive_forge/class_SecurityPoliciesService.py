from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class SecurityPoliciesService(base_api.BaseApiService):
    """Service class for the securityPolicies resource."""
    _NAME = 'securityPolicies'

    def __init__(self, client):
        super(ComputeBeta.SecurityPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def AddRule(self, request, global_params=None):
        """Inserts a rule into a security policy.

      Args:
        request: (ComputeSecurityPoliciesAddRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddRule')
        return self._RunMethod(config, request, global_params=global_params)
    AddRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.securityPolicies.addRule', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=['validateOnly'], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}/addRule', request_field='securityPolicyRule', request_type_name='ComputeSecurityPoliciesAddRuleRequest', response_type_name='Operation', supports_download=False)

    def AggregatedList(self, request, global_params=None):
        """Retrieves the list of all SecurityPolicy resources, regional and global, available to the specified project. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeSecurityPoliciesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPoliciesAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.securityPolicies.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/securityPolicies', request_field='', request_type_name='ComputeSecurityPoliciesAggregatedListRequest', response_type_name='SecurityPoliciesAggregatedList', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified policy.

      Args:
        request: (ComputeSecurityPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.securityPolicies.delete', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=['requestId'], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}', request_field='', request_type_name='ComputeSecurityPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """List all of the ordered rules present in a single specified policy.

      Args:
        request: (ComputeSecurityPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.securityPolicies.get', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=[], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}', request_field='', request_type_name='ComputeSecurityPoliciesGetRequest', response_type_name='SecurityPolicy', supports_download=False)

    def GetRule(self, request, global_params=None):
        """Gets a rule at the specified priority.

      Args:
        request: (ComputeSecurityPoliciesGetRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyRule) The response message.
      """
        config = self.GetMethodConfig('GetRule')
        return self._RunMethod(config, request, global_params=global_params)
    GetRule.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.securityPolicies.getRule', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=['priority'], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}/getRule', request_field='', request_type_name='ComputeSecurityPoliciesGetRuleRequest', response_type_name='SecurityPolicyRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new policy in the specified project using the data included in the request.

      Args:
        request: (ComputeSecurityPoliciesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.securityPolicies.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId', 'validateOnly'], relative_path='projects/{project}/global/securityPolicies', request_field='securityPolicy', request_type_name='ComputeSecurityPoliciesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """List all the policies that have been configured for the specified project.

      Args:
        request: (ComputeSecurityPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPolicyList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.securityPolicies.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/securityPolicies', request_field='', request_type_name='ComputeSecurityPoliciesListRequest', response_type_name='SecurityPolicyList', supports_download=False)

    def ListPreconfiguredExpressionSets(self, request, global_params=None):
        """Gets the current list of preconfigured Web Application Firewall (WAF) expressions.

      Args:
        request: (ComputeSecurityPoliciesListPreconfiguredExpressionSetsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SecurityPoliciesListPreconfiguredExpressionSetsResponse) The response message.
      """
        config = self.GetMethodConfig('ListPreconfiguredExpressionSets')
        return self._RunMethod(config, request, global_params=global_params)
    ListPreconfiguredExpressionSets.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.securityPolicies.listPreconfiguredExpressionSets', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/securityPolicies/listPreconfiguredExpressionSets', request_field='', request_type_name='ComputeSecurityPoliciesListPreconfiguredExpressionSetsRequest', response_type_name='SecurityPoliciesListPreconfiguredExpressionSetsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches the specified policy with the data included in the request. To clear fields in the policy, leave the fields empty and specify them in the updateMask. This cannot be used to be update the rules in the policy. Please use the per rule methods like addRule, patchRule, and removeRule instead.

      Args:
        request: (ComputeSecurityPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.securityPolicies.patch', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=['requestId', 'updateMask'], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}', request_field='securityPolicyResource', request_type_name='ComputeSecurityPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def PatchRule(self, request, global_params=None):
        """Patches a rule at the specified priority. To clear fields in the rule, leave the fields empty and specify them in the updateMask.

      Args:
        request: (ComputeSecurityPoliciesPatchRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PatchRule')
        return self._RunMethod(config, request, global_params=global_params)
    PatchRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.securityPolicies.patchRule', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=['priority', 'updateMask', 'validateOnly'], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}/patchRule', request_field='securityPolicyRule', request_type_name='ComputeSecurityPoliciesPatchRuleRequest', response_type_name='Operation', supports_download=False)

    def RemoveRule(self, request, global_params=None):
        """Deletes a rule at the specified priority.

      Args:
        request: (ComputeSecurityPoliciesRemoveRuleRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RemoveRule')
        return self._RunMethod(config, request, global_params=global_params)
    RemoveRule.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.securityPolicies.removeRule', ordered_params=['project', 'securityPolicy'], path_params=['project', 'securityPolicy'], query_params=['priority'], relative_path='projects/{project}/global/securityPolicies/{securityPolicy}/removeRule', request_field='', request_type_name='ComputeSecurityPoliciesRemoveRuleRequest', response_type_name='Operation', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on a security policy. To learn more about labels, read the Labeling Resources documentation.

      Args:
        request: (ComputeSecurityPoliciesSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.securityPolicies.setLabels', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/securityPolicies/{resource}/setLabels', request_field='globalSetLabelsRequest', request_type_name='ComputeSecurityPoliciesSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeSecurityPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.securityPolicies.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/securityPolicies/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeSecurityPoliciesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)