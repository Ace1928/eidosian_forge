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
def PromptForNameServersTransfer(api_version, domain):
    """Asks the user to provide DNS settings interactively for Transfers.

  Args:
    api_version: Cloud Domains API version to call.
    domain: Domain name corresponding to the DNS settings.

  Returns:
    A triple: (messages.DnsSettings, DnsUpdateMask, _) to be updated, or
    (None, None, _) if the user cancelled. The third value returns true when
    keeping the current DNS settings during Transfer.
  """
    domains_messages = registrations.GetMessagesModule(api_version)
    options = ['Provide Cloud DNS Managed Zone name', 'Use free name servers provided by Google Domains', 'Keep current DNS settings from current registrar']
    message = "You can provide your DNS settings in one of several ways:\nYou can specify a Cloud DNS Managed Zone name. To avoid downtime following transfer, make sure the zone is configured correctly before proceeding.\nYou can select free name servers provided by Google Domains. This blank-slate option cannot be configured before transfer.\nYou can also choose to keep the domain's DNS settings from its current registrar. Use this option only if you are sure that the domain's current DNS service will not cease upon transfer, as is often the case for DNS services provided for free by the registrar."
    cancel_option = False
    default = 2
    enable_dnssec = False
    index = console_io.PromptChoice(message=message, options=options, cancel_option=cancel_option, default=default)
    if index == 0:
        while True:
            zone = util.PromptWithValidator(validator=util.ValidateNonEmpty, error_message=' Cloud DNS Managed Zone name must not be empty.', prompt_string='Cloud DNS Managed Zone name:  ')
            try:
                name_servers, ds_records = _GetCloudDnsDetails(domains_messages, zone, domain, enable_dnssec)
            except (exceptions.Error, calliope_exceptions.HttpException) as e:
                log.status.Print(six.text_type(e))
            else:
                break
        dns_settings, update_mask = _CustomNameServers(domains_messages, name_servers, ds_records)
        return (dns_settings, update_mask, False)
    elif index == 1:
        dns_settings, update_mask = _GoogleDomainsNameServers(domains_messages, enable_dnssec)
        return (dns_settings, update_mask, False)
    else:
        return (None, None, True)