from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
class GlobalForwardingRulesService(base_api.BaseApiService):
    """Service class for the globalForwardingRules resource."""
    _NAME = 'globalForwardingRules'

    def __init__(self, client):
        super(ComputeBeta.GlobalForwardingRulesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the specified GlobalForwardingRule resource.

      Args:
        request: (ComputeGlobalForwardingRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.globalForwardingRules.delete', ordered_params=['project', 'forwardingRule'], path_params=['forwardingRule', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/forwardingRules/{forwardingRule}', request_field='', request_type_name='ComputeGlobalForwardingRulesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified GlobalForwardingRule resource. Gets a list of available forwarding rules by making a list() request.

      Args:
        request: (ComputeGlobalForwardingRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ForwardingRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalForwardingRules.get', ordered_params=['project', 'forwardingRule'], path_params=['forwardingRule', 'project'], query_params=[], relative_path='projects/{project}/global/forwardingRules/{forwardingRule}', request_field='', request_type_name='ComputeGlobalForwardingRulesGetRequest', response_type_name='ForwardingRule', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a GlobalForwardingRule resource in the specified project using the data included in the request.

      Args:
        request: (ComputeGlobalForwardingRulesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalForwardingRules.insert', ordered_params=['project'], path_params=['project'], query_params=['requestId'], relative_path='projects/{project}/global/forwardingRules', request_field='forwardingRule', request_type_name='ComputeGlobalForwardingRulesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of GlobalForwardingRule resources available to the specified project.

      Args:
        request: (ComputeGlobalForwardingRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ForwardingRuleList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.globalForwardingRules.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/global/forwardingRules', request_field='', request_type_name='ComputeGlobalForwardingRulesListRequest', response_type_name='ForwardingRuleList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified forwarding rule with the data included in the request. This method supports PATCH semantics and uses the JSON merge patch format and processing rules. Currently, you can only patch the network_tier field.

      Args:
        request: (ComputeGlobalForwardingRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='compute.globalForwardingRules.patch', ordered_params=['project', 'forwardingRule'], path_params=['forwardingRule', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/forwardingRules/{forwardingRule}', request_field='forwardingRuleResource', request_type_name='ComputeGlobalForwardingRulesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetLabels(self, request, global_params=None):
        """Sets the labels on the specified resource. To learn more about labels, read the Labeling resources documentation.

      Args:
        request: (ComputeGlobalForwardingRulesSetLabelsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetLabels')
        return self._RunMethod(config, request, global_params=global_params)
    SetLabels.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalForwardingRules.setLabels', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/forwardingRules/{resource}/setLabels', request_field='globalSetLabelsRequest', request_type_name='ComputeGlobalForwardingRulesSetLabelsRequest', response_type_name='Operation', supports_download=False)

    def SetTarget(self, request, global_params=None):
        """Changes target URL for the GlobalForwardingRule resource. The new target should be of the same type as the old target.

      Args:
        request: (ComputeGlobalForwardingRulesSetTargetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('SetTarget')
        return self._RunMethod(config, request, global_params=global_params)
    SetTarget.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalForwardingRules.setTarget', ordered_params=['project', 'forwardingRule'], path_params=['forwardingRule', 'project'], query_params=['requestId'], relative_path='projects/{project}/global/forwardingRules/{forwardingRule}/setTarget', request_field='targetReference', request_type_name='ComputeGlobalForwardingRulesSetTargetRequest', response_type_name='Operation', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource.

      Args:
        request: (ComputeGlobalForwardingRulesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.globalForwardingRules.testIamPermissions', ordered_params=['project', 'resource'], path_params=['project', 'resource'], query_params=[], relative_path='projects/{project}/global/forwardingRules/{resource}/testIamPermissions', request_field='testPermissionsRequest', request_type_name='ComputeGlobalForwardingRulesTestIamPermissionsRequest', response_type_name='TestPermissionsResponse', supports_download=False)