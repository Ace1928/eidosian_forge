from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicemanagement.v1 import servicemanagement_v1_messages as messages
class ServicesConsumersService(base_api.BaseApiService):
    """Service class for the services_consumers resource."""
    _NAME = 'services_consumers'

    def __init__(self, client):
        super(ServicemanagementV1.ServicesConsumersService, self).__init__(client)
        self._upload_configs = {}

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ServicemanagementServicesConsumersGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='servicemanagement.services.consumers.getIamPolicy', ordered_params=['servicesId', 'consumersId'], path_params=['consumersId', 'servicesId'], query_params=[], relative_path='v1/services/{servicesId}/consumers/{consumersId}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='ServicemanagementServicesConsumersGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ServicemanagementServicesConsumersSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='servicemanagement.services.consumers.setIamPolicy', ordered_params=['servicesId', 'consumersId'], path_params=['consumersId', 'servicesId'], query_params=[], relative_path='v1/services/{servicesId}/consumers/{consumersId}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ServicemanagementServicesConsumersSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ServicemanagementServicesConsumersTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='servicemanagement.services.consumers.testIamPermissions', ordered_params=['servicesId', 'consumersId'], path_params=['consumersId', 'servicesId'], query_params=[], relative_path='v1/services/{servicesId}/consumers/{consumersId}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ServicemanagementServicesConsumersTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)