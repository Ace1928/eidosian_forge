from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iap.v1beta1 import iap_v1beta1_messages as messages
class V1beta1Service(base_api.BaseApiService):
    """Service class for the v1beta1 resource."""
    _NAME = 'v1beta1'

    def __init__(self, client):
        super(IapV1beta1.V1beta1Service, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for an Identity-Aware Proxy protected resource. More information about managing access via IAP can be found at: https://cloud.google.com/iap/docs/managing-access#managing_access_via_the_api.

      Args:
        request: (IapGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}:getIamPolicy', http_method='POST', method_id='iap.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='IapGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy for an Identity-Aware Proxy protected resource. Replaces any existing policy. More information about managing access via IAP can be found at: https://cloud.google.com/iap/docs/managing-access#managing_access_via_the_api.

      Args:
        request: (IapSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}:setIamPolicy', http_method='POST', method_id='iap.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='IapSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the Identity-Aware Proxy protected resource. If the resource does not exist or the caller does not have Identity-Aware Proxy permissions a [google.rpc.Code.PERMISSION_DENIED] will be returned. More information about managing access via IAP can be found at: https://cloud.google.com/iap/docs/managing-access#managing_access_via_the_api.

      Args:
        request: (IapTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/{v1beta1Id}:testIamPermissions', http_method='POST', method_id='iap.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='IapTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)