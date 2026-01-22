from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base as base
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
class AppengineDomainsApiClient(base.AppengineApiClientBase):
    """Client used by gcloud to communicate with the App Engine API."""

    def __init__(self, client):
        base.AppengineApiClientBase.__init__(self, client)
        self._registry = resources.REGISTRY.Clone()
        self._registry.RegisterApiByName('appengine', client._VERSION)

    def DeleteDomainMapping(self, domain):
        """Deletes a domain mapping for the given application.

    Args:
      domain: str, the domain to delete.
    """
        request = self.messages.AppengineAppsDomainMappingsDeleteRequest(name=self._FormatDomainMapping(domain))
        operation = self.client.apps_domainMappings.Delete(request)
        operations_util.WaitForOperation(self.client.apps_operations, operation)

    def GetDomainMapping(self, domain):
        """Gets a domain mapping for the given application.

    Args:
      domain: str, the domain to retrieve.

    Returns:
      The retrieved DomainMapping object.
    """
        request = self.messages.AppengineAppsDomainMappingsGetRequest(name=self._FormatDomainMapping(domain))
        return self.client.apps_domainMappings.Get(request)

    def ListDomainMappings(self):
        """Lists all domain mappings for the given application.

    Returns:
      A list of DomainMapping objects.
    """
        request = self.messages.AppengineAppsDomainMappingsListRequest(parent=self._FormatApp())
        response = self.client.apps_domainMappings.List(request)
        return response.domainMappings

    def ListVerifiedDomains(self):
        """Lists all domains verified by the current user.

    Returns:
      A list of AuthorizedDomain objects.
    """
        request = self.messages.AppengineAppsAuthorizedDomainsListRequest(parent=self._FormatApp())
        response = self.client.apps_authorizedDomains.List(request)
        return response.domains

    def CreateDomainMapping(self, domain, certificate_id, management_type):
        """Creates a domain mapping for the given application.

    Args:
      domain: str, the custom domain string.
      certificate_id: str, a certificate id for the new domain.
      management_type: SslSettings.SslManagementTypeValueValuesEnum,
                       AUTOMATIC or MANUAL certificate provisioning.

    Returns:
      The created DomainMapping object.
    """
        ssl = self.messages.SslSettings(certificateId=certificate_id, sslManagementType=management_type)
        domain_mapping = self.messages.DomainMapping(id=domain, sslSettings=ssl)
        request = self.messages.AppengineAppsDomainMappingsCreateRequest(parent=self._FormatApp(), domainMapping=domain_mapping)
        operation = self.client.apps_domainMappings.Create(request)
        return operations_util.WaitForOperation(self.client.apps_operations, operation).response

    def UpdateDomainMapping(self, domain, certificate_id, no_certificate_id, management_type):
        """Updates a domain mapping for the given application.

    Args:
      domain: str, the custom domain string.
      certificate_id: str, a certificate id for the domain.
      no_certificate_id: bool, remove the certificate id from the domain.
      management_type: SslSettings.SslManagementTypeValueValuesEnum,
                       AUTOMATIC or MANUAL certificate provisioning.

    Returns:
      The updated DomainMapping object.
    """
        mask_fields = []
        if certificate_id or no_certificate_id:
            mask_fields.append('sslSettings.certificateId')
        if management_type:
            mask_fields.append('sslSettings.sslManagementType')
        ssl = self.messages.SslSettings(certificateId=certificate_id, sslManagementType=management_type)
        domain_mapping = self.messages.DomainMapping(id=domain, sslSettings=ssl)
        if not mask_fields:
            raise exceptions.MinimumArgumentException(['--[no-]certificate-id', '--no_managed_certificate'], 'Please specify at least one attribute to the domain-mapping update.')
        request = self.messages.AppengineAppsDomainMappingsPatchRequest(name=self._FormatDomainMapping(domain), domainMapping=domain_mapping, updateMask=','.join(mask_fields))
        operation = self.client.apps_domainMappings.Patch(request)
        return operations_util.WaitForOperation(self.client.apps_operations, operation).response

    def _FormatDomainMapping(self, domain):
        res = self._registry.Parse(domain, params={'appsId': self.project}, collection='appengine.apps.domainMappings')
        return res.RelativeName()