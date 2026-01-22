from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsDeploymentsService(base_api.BaseApiService):
    """Service class for the organizations_environments_deployments resource."""
    _NAME = 'organizations_environments_deployments'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a particular deployment of Api proxy or a shared flow in an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Deployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/deployments/{deploymentsId}', http_method='GET', method_id='apigee.organizations.environments.deployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsDeploymentsGetRequest', response_type_name='GoogleCloudApigeeV1Deployment', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the IAM policy on a deployment. For more information, see [Manage users, roles, and permissions using the API](https://cloud.google.com/apigee/docs/api-platform/system-administration/manage-users-roles). You must have the `apigee.deployments.getIamPolicy` permission to call this API.

      Args:
        request: (ApigeeOrganizationsEnvironmentsDeploymentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/deployments/{deploymentsId}:getIamPolicy', http_method='GET', method_id='apigee.organizations.environments.deployments.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsDeploymentsGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all deployments of API proxies or shared flows in an environment.

      Args:
        request: (ApigeeOrganizationsEnvironmentsDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/deployments', http_method='GET', method_id='apigee.organizations.environments.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['sharedFlows'], relative_path='v1/{+parent}/deployments', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsDeploymentsListRequest', response_type_name='GoogleCloudApigeeV1ListDeploymentsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the IAM policy on a deployment, if the policy already exists it will be replaced. For more information, see [Manage users, roles, and permissions using the API](https://cloud.google.com/apigee/docs/api-platform/system-administration/manage-users-roles). You must have the `apigee.deployments.setIamPolicy` permission to call this API.

      Args:
        request: (ApigeeOrganizationsEnvironmentsDeploymentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/deployments/{deploymentsId}:setIamPolicy', http_method='POST', method_id='apigee.organizations.environments.deployments.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='ApigeeOrganizationsEnvironmentsDeploymentsSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests the permissions of a user on a deployment, and returns a subset of permissions that the user has on the deployment. If the deployment does not exist, an empty permission set is returned (a NOT_FOUND error is not returned).

      Args:
        request: (ApigeeOrganizationsEnvironmentsDeploymentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/deployments/{deploymentsId}:testIamPermissions', http_method='POST', method_id='apigee.organizations.environments.deployments.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='ApigeeOrganizationsEnvironmentsDeploymentsTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)