from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v2 import iam_v2_messages as messages
class PoliciesService(base_api.BaseApiService):
    """Service class for the policies resource."""
    _NAME = 'policies'

    def __init__(self, client):
        super(IamV2.PoliciesService, self).__init__(client)
        self._upload_configs = {}

    def CreatePolicy(self, request, global_params=None):
        """Creates a policy.

      Args:
        request: (IamPoliciesCreatePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('CreatePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    CreatePolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/policies/{policiesId}/{policiesId1}', http_method='POST', method_id='iam.policies.createPolicy', ordered_params=['parent'], path_params=['parent'], query_params=['policyId'], relative_path='v2/{+parent}', request_field='googleIamV2Policy', request_type_name='IamPoliciesCreatePolicyRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a policy. This action is permanent.

      Args:
        request: (IamPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/policies/{policiesId}/{policiesId1}/{policiesId2}', http_method='DELETE', method_id='iam.policies.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v2/{+name}', request_field='', request_type_name='IamPoliciesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a policy.

      Args:
        request: (IamPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV2Policy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/policies/{policiesId}/{policiesId1}/{policiesId2}', http_method='GET', method_id='iam.policies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='IamPoliciesGetRequest', response_type_name='GoogleIamV2Policy', supports_download=False)

    def ListPolicies(self, request, global_params=None):
        """Retrieves the policies of the specified kind that are attached to a resource. The response lists only policy metadata. In particular, policy rules are omitted.

      Args:
        request: (IamPoliciesListPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV2ListPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('ListPolicies')
        return self._RunMethod(config, request, global_params=global_params)
    ListPolicies.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/policies/{policiesId}/{policiesId1}', http_method='GET', method_id='iam.policies.listPolicies', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}', request_field='', request_type_name='IamPoliciesListPoliciesRequest', response_type_name='GoogleIamV2ListPoliciesResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the specified policy. You can update only the rules and the display name for the policy. To update a policy, you should use a read-modify-write loop: 1. Use GetPolicy to read the current version of the policy. 2. Modify the policy as needed. 3. Use `UpdatePolicy` to write the updated policy. This pattern helps prevent conflicts between concurrent updates.

      Args:
        request: (GoogleIamV2Policy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/policies/{policiesId}/{policiesId1}/{policiesId2}', http_method='PUT', method_id='iam.policies.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='<request>', request_type_name='GoogleIamV2Policy', response_type_name='GoogleLongrunningOperation', supports_download=False)