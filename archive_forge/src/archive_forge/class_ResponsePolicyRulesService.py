from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
class ResponsePolicyRulesService(base_api.BaseApiService):
    """Service class for the responsePolicyRules resource."""
    _NAME = 'responsePolicyRules'

    def __init__(self, client):
        super(DnsV1alpha2.ResponsePolicyRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Response Policy Rule.

      Args:
        request: (DnsResponsePolicyRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicyRule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='dns.responsePolicyRules.create', ordered_params=['project', 'responsePolicy'], path_params=['project', 'responsePolicy'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}/rules', request_field='responsePolicyRule', request_type_name='DnsResponsePolicyRulesCreateRequest', response_type_name='ResponsePolicyRule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a previously created Response Policy Rule.

      Args:
        request: (DnsResponsePolicyRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsResponsePolicyRulesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='dns.responsePolicyRules.delete', ordered_params=['project', 'responsePolicy', 'responsePolicyRule'], path_params=['project', 'responsePolicy', 'responsePolicyRule'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}/rules/{responsePolicyRule}', request_field='', request_type_name='DnsResponsePolicyRulesDeleteRequest', response_type_name='DnsResponsePolicyRulesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Fetches the representation of an existing Response Policy Rule.

      Args:
        request: (DnsResponsePolicyRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicyRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.responsePolicyRules.get', ordered_params=['project', 'responsePolicy', 'responsePolicyRule'], path_params=['project', 'responsePolicy', 'responsePolicyRule'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}/rules/{responsePolicyRule}', request_field='', request_type_name='DnsResponsePolicyRulesGetRequest', response_type_name='ResponsePolicyRule', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerates all Response Policy Rules associated with a project.

      Args:
        request: (DnsResponsePolicyRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicyRulesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.responsePolicyRules.list', ordered_params=['project', 'responsePolicy'], path_params=['project', 'responsePolicy'], query_params=['maxResults', 'pageToken'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}/rules', request_field='', request_type_name='DnsResponsePolicyRulesListRequest', response_type_name='ResponsePolicyRulesListResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Applies a partial update to an existing Response Policy Rule.

      Args:
        request: (DnsResponsePolicyRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicyRulesPatchResponse) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='dns.responsePolicyRules.patch', ordered_params=['project', 'responsePolicy', 'responsePolicyRule'], path_params=['project', 'responsePolicy', 'responsePolicyRule'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}/rules/{responsePolicyRule}', request_field='responsePolicyRuleResource', request_type_name='DnsResponsePolicyRulesPatchRequest', response_type_name='ResponsePolicyRulesPatchResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing Response Policy Rule.

      Args:
        request: (DnsResponsePolicyRulesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResponsePolicyRulesUpdateResponse) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='dns.responsePolicyRules.update', ordered_params=['project', 'responsePolicy', 'responsePolicyRule'], path_params=['project', 'responsePolicy', 'responsePolicyRule'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/responsePolicies/{responsePolicy}/rules/{responsePolicyRule}', request_field='responsePolicyRuleResource', request_type_name='DnsResponsePolicyRulesUpdateRequest', response_type_name='ResponsePolicyRulesUpdateResponse', supports_download=False)