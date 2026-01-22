from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
class ProjectsLocationsRegistrationsService(base_api.BaseApiService):
    """Service class for the projects_locations_registrations resource."""
    _NAME = 'projects_locations_registrations'

    def __init__(self, client):
        super(DomainsV1alpha2.ProjectsLocationsRegistrationsService, self).__init__(client)
        self._upload_configs = {}

    def ConfigureContactSettings(self, request, global_params=None):
        """Updates a `Registration`'s contact settings. Some changes require confirmation by the domain's registrant contact . Caution: Please consider carefully any changes to contact privacy settings when changing from `REDACTED_CONTACT_DATA` to `PUBLIC_CONTACT_DATA.` There may be a delay in reflecting updates you make to registrant contact information such that any changes you make to contact privacy (including from `REDACTED_CONTACT_DATA` to `PUBLIC_CONTACT_DATA`) will be applied without delay but changes to registrant contact information may take a limited time to be publicized. This means that changes to contact privacy from `REDACTED_CONTACT_DATA` to `PUBLIC_CONTACT_DATA` may make the previous registrant contact data public until the modified registrant contact details are published.

      Args:
        request: (DomainsProjectsLocationsRegistrationsConfigureContactSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ConfigureContactSettings')
        return self._RunMethod(config, request, global_params=global_params)
    ConfigureContactSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:configureContactSettings', http_method='POST', method_id='domains.projects.locations.registrations.configureContactSettings', ordered_params=['registration'], path_params=['registration'], query_params=[], relative_path='v1alpha2/{+registration}:configureContactSettings', request_field='configureContactSettingsRequest', request_type_name='DomainsProjectsLocationsRegistrationsConfigureContactSettingsRequest', response_type_name='Operation', supports_download=False)

    def ConfigureDnsSettings(self, request, global_params=None):
        """Updates a `Registration`'s DNS settings.

      Args:
        request: (DomainsProjectsLocationsRegistrationsConfigureDnsSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ConfigureDnsSettings')
        return self._RunMethod(config, request, global_params=global_params)
    ConfigureDnsSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:configureDnsSettings', http_method='POST', method_id='domains.projects.locations.registrations.configureDnsSettings', ordered_params=['registration'], path_params=['registration'], query_params=[], relative_path='v1alpha2/{+registration}:configureDnsSettings', request_field='configureDnsSettingsRequest', request_type_name='DomainsProjectsLocationsRegistrationsConfigureDnsSettingsRequest', response_type_name='Operation', supports_download=False)

    def ConfigureManagementSettings(self, request, global_params=None):
        """Updates a `Registration`'s management settings.

      Args:
        request: (DomainsProjectsLocationsRegistrationsConfigureManagementSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ConfigureManagementSettings')
        return self._RunMethod(config, request, global_params=global_params)
    ConfigureManagementSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:configureManagementSettings', http_method='POST', method_id='domains.projects.locations.registrations.configureManagementSettings', ordered_params=['registration'], path_params=['registration'], query_params=[], relative_path='v1alpha2/{+registration}:configureManagementSettings', request_field='configureManagementSettingsRequest', request_type_name='DomainsProjectsLocationsRegistrationsConfigureManagementSettingsRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `Registration` resource. This method works on any `Registration` resource using [Subscription or Commitment billing](/domains/pricing#billing-models), provided that the resource was created at least 1 day in the past. When an active registration is successfully deleted, you can continue to use the domain in [Google Domains](https://domains.google/) until it expires. The calling user becomes the domain's sole owner in Google Domains, and permissions for the domain are subsequently managed there. The domain does not renew automatically unless the new owner sets up billing in Google Domains. After January 2024 you will only be able to delete `Registration` resources when `state` is one of: `EXPORTED`, `EXPIRED`,`REGISTRATION_FAILED` or `TRANSFER_FAILED`. See [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) for more details.

      Args:
        request: (DomainsProjectsLocationsRegistrationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}', http_method='DELETE', method_id='domains.projects.locations.registrations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Export(self, request, global_params=None):
        """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Exports a `Registration` resource, such that it is no longer managed by Cloud Domains. When an active domain is successfully exported, you can continue to use the domain in [Google Domains](https://domains.google/) until it expires. The calling user becomes the domain's sole owner in Google Domains, and permissions for the domain are subsequently managed there. The domain does not renew automatically unless the new owner sets up billing in Google Domains.

      Args:
        request: (DomainsProjectsLocationsRegistrationsExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:export', http_method='POST', method_id='domains.projects.locations.registrations.export', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:export', request_field='exportRegistrationRequest', request_type_name='DomainsProjectsLocationsRegistrationsExportRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a `Registration` resource.

      Args:
        request: (DomainsProjectsLocationsRegistrationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Registration) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}', http_method='GET', method_id='domains.projects.locations.registrations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsGetRequest', response_type_name='Registration', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DomainsProjectsLocationsRegistrationsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:getIamPolicy', http_method='GET', method_id='domains.projects.locations.registrations.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def Import(self, request, global_params=None):
        """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Imports a domain name from [Google Domains](https://domains.google/) for use in Cloud Domains. To transfer a domain from another registrar, use the `TransferDomain` method instead. Since individual users can own domains in Google Domains, the calling user must have ownership permission on the domain.

      Args:
        request: (DomainsProjectsLocationsRegistrationsImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:import', http_method='POST', method_id='domains.projects.locations.registrations.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/registrations:import', request_field='importDomainRequest', request_type_name='DomainsProjectsLocationsRegistrationsImportRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the `Registration` resources in a project.

      Args:
        request: (DomainsProjectsLocationsRegistrationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRegistrationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations', http_method='GET', method_id='domains.projects.locations.registrations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/registrations', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsListRequest', response_type_name='ListRegistrationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates select fields of a `Registration` resource, notably `labels`. To update other fields, use the appropriate custom update method: * To update management settings, see `ConfigureManagementSettings` * To update DNS configuration, see `ConfigureDnsSettings` * To update contact information, see `ConfigureContactSettings`.

      Args:
        request: (DomainsProjectsLocationsRegistrationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}', http_method='PATCH', method_id='domains.projects.locations.registrations.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='registration', request_type_name='DomainsProjectsLocationsRegistrationsPatchRequest', response_type_name='Operation', supports_download=False)

    def Register(self, request, global_params=None):
        """Registers a new domain name and creates a corresponding `Registration` resource. Call `RetrieveRegisterParameters` first to check availability of the domain name and determine parameters like price that are needed to build a call to this method. A successful call creates a `Registration` resource in state `REGISTRATION_PENDING`, which resolves to `ACTIVE` within 1-2 minutes, indicating that the domain was successfully registered. If the resource ends up in state `REGISTRATION_FAILED`, it indicates that the domain was not registered successfully, and you can safely delete the resource and retry registration.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRegisterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Register')
        return self._RunMethod(config, request, global_params=global_params)
    Register.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:register', http_method='POST', method_id='domains.projects.locations.registrations.register', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/registrations:register', request_field='registerDomainRequest', request_type_name='DomainsProjectsLocationsRegistrationsRegisterRequest', response_type_name='Operation', supports_download=False)

    def ResetAuthorizationCode(self, request, global_params=None):
        """Resets the authorization code of the `Registration` to a new random string. You can call this method only after 60 days have elapsed since the initial domain registration.

      Args:
        request: (DomainsProjectsLocationsRegistrationsResetAuthorizationCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizationCode) The response message.
      """
        config = self.GetMethodConfig('ResetAuthorizationCode')
        return self._RunMethod(config, request, global_params=global_params)
    ResetAuthorizationCode.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:resetAuthorizationCode', http_method='POST', method_id='domains.projects.locations.registrations.resetAuthorizationCode', ordered_params=['registration'], path_params=['registration'], query_params=[], relative_path='v1alpha2/{+registration}:resetAuthorizationCode', request_field='resetAuthorizationCodeRequest', request_type_name='DomainsProjectsLocationsRegistrationsResetAuthorizationCodeRequest', response_type_name='AuthorizationCode', supports_download=False)

    def RetrieveAuthorizationCode(self, request, global_params=None):
        """Gets the authorization code of the `Registration` for the purpose of transferring the domain to another registrar. You can call this method only after 60 days have elapsed since the initial domain registration.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveAuthorizationCodeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AuthorizationCode) The response message.
      """
        config = self.GetMethodConfig('RetrieveAuthorizationCode')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveAuthorizationCode.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:retrieveAuthorizationCode', http_method='GET', method_id='domains.projects.locations.registrations.retrieveAuthorizationCode', ordered_params=['registration'], path_params=['registration'], query_params=[], relative_path='v1alpha2/{+registration}:retrieveAuthorizationCode', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsRetrieveAuthorizationCodeRequest', response_type_name='AuthorizationCode', supports_download=False)

    def RetrieveImportableDomains(self, request, global_params=None):
        """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Lists domain names from [Google Domains](https://domains.google/) that can be imported to Cloud Domains using the `ImportDomain` method. Since individual users can own domains in Google Domains, the list of domains returned depends on the individual user making the call. Domains already managed by Cloud Domains are not returned.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveImportableDomainsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetrieveImportableDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('RetrieveImportableDomains')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveImportableDomains.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:retrieveImportableDomains', http_method='GET', method_id='domains.projects.locations.registrations.retrieveImportableDomains', ordered_params=['location'], path_params=['location'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/{+location}/registrations:retrieveImportableDomains', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsRetrieveImportableDomainsRequest', response_type_name='RetrieveImportableDomainsResponse', supports_download=False)

    def RetrieveRegisterParameters(self, request, global_params=None):
        """Gets parameters needed to register a new domain name, including price and up-to-date availability. Use the returned values to call `RegisterDomain`.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveRegisterParametersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetrieveRegisterParametersResponse) The response message.
      """
        config = self.GetMethodConfig('RetrieveRegisterParameters')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveRegisterParameters.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:retrieveRegisterParameters', http_method='GET', method_id='domains.projects.locations.registrations.retrieveRegisterParameters', ordered_params=['location'], path_params=['location'], query_params=['domainName'], relative_path='v1alpha2/{+location}/registrations:retrieveRegisterParameters', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsRetrieveRegisterParametersRequest', response_type_name='RetrieveRegisterParametersResponse', supports_download=False)

    def RetrieveTransferParameters(self, request, global_params=None):
        """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Gets parameters needed to transfer a domain name from another registrar to Cloud Domains. For domains already managed by [Google Domains](https://domains.google/), use `ImportDomain` instead. Use the returned values to call `TransferDomain`.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveTransferParametersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetrieveTransferParametersResponse) The response message.
      """
        config = self.GetMethodConfig('RetrieveTransferParameters')
        return self._RunMethod(config, request, global_params=global_params)
    RetrieveTransferParameters.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:retrieveTransferParameters', http_method='GET', method_id='domains.projects.locations.registrations.retrieveTransferParameters', ordered_params=['location'], path_params=['location'], query_params=['domainName'], relative_path='v1alpha2/{+location}/registrations:retrieveTransferParameters', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsRetrieveTransferParametersRequest', response_type_name='RetrieveTransferParametersResponse', supports_download=False)

    def SearchDomains(self, request, global_params=None):
        """Searches for available domain names similar to the provided query. Availability results from this method are approximate; call `RetrieveRegisterParameters` on a domain before registering to confirm availability.

      Args:
        request: (DomainsProjectsLocationsRegistrationsSearchDomainsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchDomains')
        return self._RunMethod(config, request, global_params=global_params)
    SearchDomains.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:searchDomains', http_method='GET', method_id='domains.projects.locations.registrations.searchDomains', ordered_params=['location'], path_params=['location'], query_params=['query'], relative_path='v1alpha2/{+location}/registrations:searchDomains', request_field='', request_type_name='DomainsProjectsLocationsRegistrationsSearchDomainsRequest', response_type_name='SearchDomainsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (DomainsProjectsLocationsRegistrationsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:setIamPolicy', http_method='POST', method_id='domains.projects.locations.registrations.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DomainsProjectsLocationsRegistrationsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DomainsProjectsLocationsRegistrationsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations/{registrationsId}:testIamPermissions', http_method='POST', method_id='domains.projects.locations.registrations.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DomainsProjectsLocationsRegistrationsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Transfer(self, request, global_params=None):
        """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Transfers a domain name from another registrar to Cloud Domains. For domains already managed by [Google Domains](https://domains.google/), use `ImportDomain` instead. Before calling this method, go to the domain's current registrar to unlock the domain for transfer and retrieve the domain's transfer authorization code. Then call `RetrieveTransferParameters` to confirm that the domain is unlocked and to get values needed to build a call to this method. A successful call creates a `Registration` resource in state `TRANSFER_PENDING`. It can take several days to complete the transfer process. The registrant can often speed up this process by approving the transfer through the current registrar, either by clicking a link in an email from the registrar or by visiting the registrar's website. A few minutes after transfer approval, the resource transitions to state `ACTIVE`, indicating that the transfer was successful. If the transfer is rejected or the request expires without being approved, the resource can end up in state `TRANSFER_FAILED`. If transfer fails, you can safely delete the resource and retry the transfer.

      Args:
        request: (DomainsProjectsLocationsRegistrationsTransferRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Transfer')
        return self._RunMethod(config, request, global_params=global_params)
    Transfer.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/registrations:transfer', http_method='POST', method_id='domains.projects.locations.registrations.transfer', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/registrations:transfer', request_field='transferDomainRequest', request_type_name='DomainsProjectsLocationsRegistrationsTransferRequest', response_type_name='Operation', supports_download=False)