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
def ParseDNSSettings(api_version, name_servers, cloud_dns_zone, use_google_domains_dns, dns_settings_from_file, domain, enable_dnssec=True, dns_settings=None):
    """Parses DNS settings from a flag.

  At most one of the arguments (except domain) should be non-empty.

  Args:
    api_version: Cloud Domains API version to call.
    name_servers: List of name servers
    cloud_dns_zone: Cloud DNS Zone name
    use_google_domains_dns: Information that Google Domains name servers should
      be used.
    dns_settings_from_file: Path to a yaml file with dns_settings.
    domain: Domain name corresponding to the DNS settings.
    enable_dnssec: Enable DNSSEC for Google Domains name servers or Cloud DNS
      Zone.
    dns_settings: Current DNS settings. Used during Configure DNS only.

  Returns:
    A pair: (messages.DnsSettings, DnsUpdateMask) to be updated, or (None, None)
    if all the arguments are empty.
  """
    domains_messages = registrations.GetMessagesModule(api_version)
    if name_servers is not None:
        return _CustomNameServers(domains_messages, name_servers)
    if cloud_dns_zone is not None:
        nameservers, ds_records = _GetCloudDnsDetails(domains_messages, cloud_dns_zone, domain, enable_dnssec)
        return _CustomNameServers(domains_messages, nameservers, ds_records)
    if use_google_domains_dns:
        return _GoogleDomainsNameServers(domains_messages, enable_dnssec)
    if dns_settings_from_file is not None:
        return _ParseDnsSettingsFromFile(domains_messages, dns_settings_from_file)
    if dns_settings is not None and (not enable_dnssec):
        return _DisableDnssec(domains_messages, dns_settings)
    return (None, None)