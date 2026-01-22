from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
def ConfigureDNS(self, registration_ref, dns_settings, updated, validate_only):
    """Calls ConfigureDNSSettings method.

    Args:
      registration_ref: Registration resource reference.
      dns_settings: New DNS Settings.
      updated: dns_util.DnsUpdateMask object representing an update mask.
      validate_only: validate_only flag.

    Returns:
      Long Running Operation reference.
    """
    updated_list = []
    if updated.glue_records:
        updated_list += ['glue_records']
    if updated.google_domains_dnssec:
        if updated.name_servers:
            updated_list += ['google_domains_dns']
        else:
            updated_list += ['google_domains_dns.ds_state']
    if updated.custom_dnssec:
        if updated.name_servers:
            updated_list += ['custom_dns']
        else:
            updated_list += ['custom_dns.ds_records']
    update_mask = ','.join(updated_list)
    req = self.messages.DomainsProjectsLocationsRegistrationsConfigureDnsSettingsRequest(registration=registration_ref.RelativeName(), configureDnsSettingsRequest=self.messages.ConfigureDnsSettingsRequest(dnsSettings=dns_settings, updateMask=update_mask, validateOnly=validate_only))
    return self._service.ConfigureDnsSettings(req)