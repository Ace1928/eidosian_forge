from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsEnvironmentsSecurityActionsService(base_api.BaseApiService):
    """Service class for the organizations_environments_securityActions resource."""
    _NAME = 'organizations_environments_securityActions'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsEnvironmentsSecurityActionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """CreateSecurityAction creates a SecurityAction.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityActionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityAction) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActions', http_method='POST', method_id='apigee.organizations.environments.securityActions.create', ordered_params=['parent'], path_params=['parent'], query_params=['securityActionId'], relative_path='v1/{+parent}/securityActions', request_field='googleCloudApigeeV1SecurityAction', request_type_name='ApigeeOrganizationsEnvironmentsSecurityActionsCreateRequest', response_type_name='GoogleCloudApigeeV1SecurityAction', supports_download=False)

    def Disable(self, request, global_params=None):
        """Disable a SecurityAction. The `state` of the SecurityAction after disabling is `DISABLED`. `DisableSecurityAction` can be called on SecurityActions in the state `ENABLED`; SecurityActions in a different state (including `DISABLED`) return an error.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityActionsDisableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityAction) The response message.
      """
        config = self.GetMethodConfig('Disable')
        return self._RunMethod(config, request, global_params=global_params)
    Disable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActions/{securityActionsId}:disable', http_method='POST', method_id='apigee.organizations.environments.securityActions.disable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:disable', request_field='googleCloudApigeeV1DisableSecurityActionRequest', request_type_name='ApigeeOrganizationsEnvironmentsSecurityActionsDisableRequest', response_type_name='GoogleCloudApigeeV1SecurityAction', supports_download=False)

    def Enable(self, request, global_params=None):
        """Enable a SecurityAction. The `state` of the SecurityAction after enabling is `ENABLED`. `EnableSecurityAction` can be called on SecurityActions in the state `DISABLED`; SecurityActions in a different state (including `ENABLED) return an error.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityActionsEnableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityAction) The response message.
      """
        config = self.GetMethodConfig('Enable')
        return self._RunMethod(config, request, global_params=global_params)
    Enable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActions/{securityActionsId}:enable', http_method='POST', method_id='apigee.organizations.environments.securityActions.enable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:enable', request_field='googleCloudApigeeV1EnableSecurityActionRequest', request_type_name='ApigeeOrganizationsEnvironmentsSecurityActionsEnableRequest', response_type_name='GoogleCloudApigeeV1SecurityAction', supports_download=False)

    def Get(self, request, global_params=None):
        """Get a SecurityAction by name.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityActionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityAction) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActions/{securityActionsId}', http_method='GET', method_id='apigee.organizations.environments.securityActions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityActionsGetRequest', response_type_name='GoogleCloudApigeeV1SecurityAction', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of SecurityActions. This returns both enabled and disabled actions.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityActionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ListSecurityActionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/environments/{environmentsId}/securityActions', http_method='GET', method_id='apigee.organizations.environments.securityActions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/securityActions', request_field='', request_type_name='ApigeeOrganizationsEnvironmentsSecurityActionsListRequest', response_type_name='GoogleCloudApigeeV1ListSecurityActionsResponse', supports_download=False)