from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.beyondcorp.v1alpha import beyondcorp_v1alpha_messages as messages
class OrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesService(base_api.BaseApiService):
    """Service class for the organizations_locations_global_partnerTenants_browserDlpRules resource."""
    _NAME = 'organizations_locations_global_partnerTenants_browserDlpRules'

    def __init__(self, client):
        super(BeyondcorpV1alpha.OrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new BrowserDlpRule in a given organization and PartnerTenant.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules', http_method='POST', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1alpha/{+parent}/browserDlpRules', request_field='googleCloudBeyondcorpPartnerservicesV1alphaBrowserDlpRule', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing BrowserDlpRule.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules/{browserDlpRulesId}', http_method='DELETE', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single BrowserDlpRule.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpPartnerservicesV1alphaBrowserDlpRule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules/{browserDlpRulesId}', http_method='GET', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesGetRequest', response_type_name='GoogleCloudBeyondcorpPartnerservicesV1alphaBrowserDlpRule', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules/{browserDlpRulesId}:getIamPolicy', http_method='GET', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha/{+resource}:getIamPolicy', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesGetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists BrowserDlpRules for PartnerTenant in a given organization.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudBeyondcorpPartnerservicesV1alphaListBrowserDlpRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules', http_method='GET', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha/{+parent}/browserDlpRules', request_field='', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesListRequest', response_type_name='GoogleCloudBeyondcorpPartnerservicesV1alphaListBrowserDlpRulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an existing BrowserDlpRule in a given organization and PartnerTenant.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules/{browserDlpRulesId}', http_method='PATCH', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='googleCloudBeyondcorpPartnerservicesV1alphaBrowserDlpRule', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules/{browserDlpRulesId}:setIamPolicy', http_method='POST', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:setIamPolicy', request_field='googleIamV1SetIamPolicyRequest', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesSetIamPolicyRequest', response_type_name='GoogleIamV1Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV1TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/organizations/{organizationsId}/locations/global/partnerTenants/{partnerTenantsId}/browserDlpRules/{browserDlpRulesId}:testIamPermissions', http_method='POST', method_id='beyondcorp.organizations.locations.global.partnerTenants.browserDlpRules.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha/{+resource}:testIamPermissions', request_field='googleIamV1TestIamPermissionsRequest', request_type_name='BeyondcorpOrganizationsLocationsGlobalPartnerTenantsBrowserDlpRulesTestIamPermissionsRequest', response_type_name='GoogleIamV1TestIamPermissionsResponse', supports_download=False)