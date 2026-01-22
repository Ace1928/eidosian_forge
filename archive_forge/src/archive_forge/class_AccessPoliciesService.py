from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
class AccessPoliciesService(base_api.BaseApiService):
    """Service class for the accessPolicies resource."""
    _NAME = 'accessPolicies'

    def __init__(self, client):
        super(AccesscontextmanagerV1alpha.AccessPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an access policy. This method fails if the organization already has an access policy. The long-running operation has a successful status after the access policy propagates to long-lasting storage. Syntactic and basic semantic errors are returned in `metadata` as a BadRequest proto.

      Args:
        request: (AccessPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='accesscontextmanager.accessPolicies.create', ordered_params=[], path_params=[], query_params=[], relative_path='v1alpha/accessPolicies', request_field='<request>', request_type_name='AccessPolicy', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an access policy based on the resource name. The long-running operation has a successful status after the access policy is removed from long-lasting storage.

      Args:
        request: (AccesscontextmanagerAccessPoliciesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}', http_method='DELETE', method_id='accesscontextmanager.accessPolicies.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns an access policy based on the name.

      Args:
        request: (AccesscontextmanagerAccessPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AccessPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}', http_method='GET', method_id='accesscontextmanager.accessPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesGetRequest', response_type_name='AccessPolicy', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy for the specified Access Context Manager access policy.

      Args:
        request: (AccesscontextmanagerAccessPoliciesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}:getIamPolicy', http_method='POST', method_id='accesscontextmanager.accessPolicies.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='AccesscontextmanagerAccessPoliciesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all access policies in an organization.

      Args:
        request: (AccesscontextmanagerAccessPoliciesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAccessPoliciesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='accesscontextmanager.accessPolicies.list', ordered_params=[], path_params=[], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v1alpha/accessPolicies', request_field='', request_type_name='AccesscontextmanagerAccessPoliciesListRequest', response_type_name='ListAccessPoliciesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an access policy. The long-running operation from this RPC has a successful status after the changes to the access policy propagate to long-lasting storage.

      Args:
        request: (AccesscontextmanagerAccessPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}', http_method='PATCH', method_id='accesscontextmanager.accessPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='accessPolicy', request_type_name='AccesscontextmanagerAccessPoliciesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy for the specified Access Context Manager access policy. This method replaces the existing IAM policy on the access policy. The IAM policy controls the set of users who can perform specific operations on the Access Context Manager access policy.

      Args:
        request: (AccesscontextmanagerAccessPoliciesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}:setIamPolicy', http_method='POST', method_id='accesscontextmanager.accessPolicies.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='AccesscontextmanagerAccessPoliciesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the IAM permissions that the caller has on the specified Access Context Manager resource. The resource can be an AccessPolicy, AccessLevel, or ServicePerimeter. This method does not support other resources.

      Args:
        request: (AccesscontextmanagerAccessPoliciesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/accessPolicies/{accessPoliciesId}:testIamPermissions', http_method='POST', method_id='accesscontextmanager.accessPolicies.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='AccesscontextmanagerAccessPoliciesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)