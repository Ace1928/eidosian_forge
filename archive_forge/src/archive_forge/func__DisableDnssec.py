from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.dns import util as dns_api_util
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.domains import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import six
def _DisableDnssec(domains_messages, dns_settings):
    """Returns DNS settings (and update mask) with DNSSEC disabled."""
    if dns_settings is None:
        return (None, None)
    if dns_settings.googleDomainsDns is not None:
        updated_dns_settings = domains_messages.DnsSettings(googleDomainsDns=domains_messages.GoogleDomainsDns(dsState=domains_messages.GoogleDomainsDns.DsStateValueValuesEnum.DS_RECORDS_UNPUBLISHED))
        update_mask = DnsUpdateMask(google_domains_dnssec=True)
    elif dns_settings.customDns is not None:
        updated_dns_settings = domains_messages.DnsSettings(customDns=domains_messages.CustomDns(dsRecords=[]))
        update_mask = DnsUpdateMask(custom_dnssec=True)
    else:
        return (None, None)
    return (updated_dns_settings, update_mask)