from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsService(base_api.BaseApiService):
    """Service class for the organizations_environments resource."""
    _NAME = 'organizations_environments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an environment in an organization.

      Args:
        request: (ApigeeOrganizationsEnvironmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments', http_method='POST', method_id='apigee.organizations.environments.create', ordered_params=['parent'], path_params=['parent'], query_params=['name'], relative_path='v1/{+parent}/environments', request_field='googleCloudApigeeV1Environment', request_type_name='ApigeeOrganizationsEnvironmentsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an environment from an organization. **Warning: You must delete all key value maps and key value entries before you delete an environment.** Otherwise, if you re-create the environment the key value map entry operations will encounter encryption/decryption discrepancies.

      Args:
        request: (ApigeeOrganizationsEnvironmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}', http_method='DELETE', method_id='apigee.organizations.environments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets environment details.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Environment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}', http_method='GET', method_id='apigee.organizations.environments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetRequest', response_type_name='GoogleCloudApigeeV1Environment', supports_download=False)

    def GetAddonsConfig(self, request, global_params=None):
        """Gets the add-ons config of an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetAddonsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1AddonsConfig) The response message.
      """
        config = self.GetMethodConfig('GetAddonsConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetAddonsConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/addonsConfig', http_method='GET', method_id='apigee.organizations.environments.getAddonsConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetAddonsConfigRequest', response_type_name='GoogleCloudApigeeV1AddonsConfig', supports_download=False)

    def GetApiSecurityRuntimeConfig(self, request, global_params=None):
        """Gets the API Security runtime configuration for an environment. This named ApiSecurityRuntimeConfig to prevent conflicts with ApiSecurityConfig from addon config.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetApiSecurityRuntimeConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ApiSecurityRuntimeConfig) The response message.
      """
        config = self.GetMethodConfig('GetApiSecurityRuntimeConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetApiSecurityRuntimeConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/apiSecurityRuntimeConfig', http_method='GET', method_id='apigee.organizations.environments.getApiSecurityRuntimeConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetApiSecurityRuntimeConfigRequest', response_type_name='GoogleCloudApigeeV1ApiSecurityRuntimeConfig', supports_download=False)

    def GetDebugmask(self, request, global_params=None):
        """Gets the debug mask singleton resource for an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetDebugmaskRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugMask) The response message.
      """
        config = self.GetMethodConfig('GetDebugmask')
        return self._RunMethod(config, request, global_params=global_params)
    GetDebugmask.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/debugmask', http_method='GET', method_id='apigee.organizations.environments.getDebugmask', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetDebugmaskRequest', response_type_name='GoogleCloudApigeeV1DebugMask', supports_download=False)

    def GetDeployedConfig(self, request, global_params=None):
        """Gets the deployed configuration for an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetDeployedConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1EnvironmentConfig) The response message.
      """
        config = self.GetMethodConfig('GetDeployedConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetDeployedConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/deployedConfig', http_method='GET', method_id='apigee.organizations.environments.getDeployedConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetDeployedConfigRequest', response_type_name='GoogleCloudApigeeV1EnvironmentConfig', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy on an environment. For more information, see [Manage users, roles, and permissions using the API](https://cloud.google.com/apigee/docs/api-platform/system-administration/manage-users-roles). You must have the `apigee.environments.getIamPolicy` permission to call this API.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}:getIamPolicy', http_method='GET', method_id='apigee.organizations.environments.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def GetSecurityActionsConfig(self, request, global_params=None):
        """GetSecurityActionConfig returns the current SecurityActions configuration.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetSecurityActionsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityActionsConfig) The response message.
      """
        config = self.GetMethodConfig('GetSecurityActionsConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetSecurityActionsConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActionsConfig', http_method='GET', method_id='apigee.organizations.environments.getSecurityActionsConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetSecurityActionsConfigRequest', response_type_name='GoogleCloudApigeeV1SecurityActionsConfig', supports_download=False)

    def GetTraceConfig(self, request, global_params=None):
        """Get distributed trace configuration in an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsGetTraceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfig) The response message.
      """
        config = self.GetMethodConfig('GetTraceConfig')
        return self._RunMethod(config, request, global_params=global_params)
    GetTraceConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig', http_method='GET', method_id='apigee.organizations.environments.getTraceConfig', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsGetTraceConfigRequest', response_type_name='GoogleCloudApigeeV1TraceConfig', supports_download=False)

    def ModifyEnvironment(self, request, global_params=None):
        """Updates properties for an Apigee environment with patch semantics using a field mask. **Note:** Not supported for Apigee hybrid.

      Args:
        request: (ApigeeOrganizationsEnvironmentsModifyEnvironmentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('ModifyEnvironment')
        return self._RunMethod(config, request, global_params=global_params)
    ModifyEnvironment.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}', http_method='PATCH', method_id='apigee.organizations.environments.modifyEnvironment', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1Environment', request_type_name='ApigeeOrganizationsEnvironmentsModifyEnvironmentRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy on an environment, if the policy already exists it will be replaced. For more information, see [Manage users, roles, and permissions using the API](https://cloud.google.com/apigee/docs/api-platform/system-administration/manage-users-roles). You must have the `apigee.environments.setIamPolicy` permission to call this API.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}:setIamPolicy', http_method='POST', method_id='apigee.organizations.environments.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='ApigeeOrganizationsEnvironmentsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def Subscribe(self, request, global_params=None):
        """Creates a subscription for the environment's Pub/Sub topic. The server will assign a random name for this subscription. The "name" and "push_config" must *not* be specified.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSubscribeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Subscription) The response message.
      """
        config = self.GetMethodConfig('Subscribe')
        return self._RunMethod(config, request, global_params=global_params)
    Subscribe.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}:subscribe', http_method='POST', method_id='apigee.organizations.environments.subscribe', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:subscribe', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSubscribeRequest', response_type_name='GoogleCloudApigeeV1Subscription', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests the permissions of a user on an environment, and returns a subset of permissions that the user has on the environment. If the environment does not exist, an empty permission set is returned (a NOT_FOUND error is not returned).

      Args:
        request: (ApigeeOrganizationsEnvironmentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}:testIamPermissions', http_method='POST', method_id='apigee.organizations.environments.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='ApigeeOrganizationsEnvironmentsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)

    def Unsubscribe(self, request, global_params=None):
        """Deletes a subscription for the environment's Pub/Sub topic.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUnsubscribeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Unsubscribe')
        return self._RunMethod(config, request, global_params=global_params)
    Unsubscribe.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}:unsubscribe', http_method='POST', method_id='apigee.organizations.environments.unsubscribe', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:unsubscribe', request_field='googleCloudApigeeV1Subscription', request_type_name='ApigeeOrganizationsEnvironmentsUnsubscribeRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing environment. When updating properties, you must pass all existing properties to the API, even if they are not being changed. If you omit properties from the payload, the properties are removed. To get the current list of properties for the environment, use the [Get Environment API](get). **Note**: Both `PUT` and `POST` methods are supported for updating an existing environment.

      Args:
        request: (GoogleCloudApigeeV1Environment) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Environment) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}', http_method='PUT', method_id='apigee.organizations.environments.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1Environment', response_type_name='GoogleCloudApigeeV1Environment', supports_download=False)

    def UpdateDebugmask(self, request, global_params=None):
        """Updates the debug mask singleton resource for an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUpdateDebugmaskRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1DebugMask) The response message.
      """
        config = self.GetMethodConfig('UpdateDebugmask')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDebugmask.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/debugmask', http_method='PATCH', method_id='apigee.organizations.environments.updateDebugmask', ordered_params=['name'], path_params=['name'], query_params=['replaceRepeatedFields', 'updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1DebugMask', request_type_name='ApigeeOrganizationsEnvironmentsUpdateDebugmaskRequest', response_type_name='GoogleCloudApigeeV1DebugMask', supports_download=False)

    def UpdateEnvironment(self, request, global_params=None):
        """Updates an existing environment. When updating properties, you must pass all existing properties to the API, even if they are not being changed. If you omit properties from the payload, the properties are removed. To get the current list of properties for the environment, use the [Get Environment API](get). **Note**: Both `PUT` and `POST` methods are supported for updating an existing environment.

      Args:
        request: (GoogleCloudApigeeV1Environment) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Environment) The response message.
      """
        config = self.GetMethodConfig('UpdateEnvironment')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateEnvironment.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}', http_method='POST', method_id='apigee.organizations.environments.updateEnvironment', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='GoogleCloudApigeeV1Environment', response_type_name='GoogleCloudApigeeV1Environment', supports_download=False)

    def UpdateSecurityActionsConfig(self, request, global_params=None):
        """UpdateSecurityActionConfig updates the current SecurityActions configuration. This method is used to enable/disable the feature at the environment level.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUpdateSecurityActionsConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityActionsConfig) The response message.
      """
        config = self.GetMethodConfig('UpdateSecurityActionsConfig')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSecurityActionsConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActionsConfig', http_method='PATCH', method_id='apigee.organizations.environments.updateSecurityActionsConfig', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1SecurityActionsConfig', request_type_name='ApigeeOrganizationsEnvironmentsUpdateSecurityActionsConfigRequest', response_type_name='GoogleCloudApigeeV1SecurityActionsConfig', supports_download=False)

    def UpdateTraceConfig(self, request, global_params=None):
        """Updates the trace configurations in an environment. Note that the repeated fields have replace semantics when included in the field mask and that they will be overwritten by the value of the fields in the request body.

      Args:
        request: (ApigeeOrganizationsEnvironmentsUpdateTraceConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1TraceConfig) The response message.
      """
        config = self.GetMethodConfig('UpdateTraceConfig')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateTraceConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/traceConfig', http_method='PATCH', method_id='apigee.organizations.environments.updateTraceConfig', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleCloudApigeeV1TraceConfig', request_type_name='ApigeeOrganizationsEnvironmentsUpdateTraceConfigRequest', response_type_name='GoogleCloudApigeeV1TraceConfig', supports_download=False)