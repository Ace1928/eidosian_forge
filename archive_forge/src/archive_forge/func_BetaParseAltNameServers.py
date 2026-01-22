from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def BetaParseAltNameServers(version, server_list=None, private_server_list=None):
    """Parses list of alternative nameservers into AlternativeNameServerConfig.

  Args:
    version: (str) A string indicating the version of the API to be used, should
      be one of 'v1beta2' or 'v1alpha2'. This function will be moved after
      promoting v6 address to GA.
    server_list: (Sequence) List of IP addresses to use as forwarding targets
      for the DNS Managed Zone that uses default forwarding logic.
    private_server_list: (Sequence) List of IP addresses to use as forwarding
      targets for the DNS Managed Zone that always uses the private VPC path.

  Returns:
    A messages.PolicyAlternativeNameServerConfig instance populated from the
    given command line arguments.Only the not none server list will be parsed
    and
    an empty list will be returned if both are none.
  """
    if not server_list and (not private_server_list):
        return None
    messages = GetMessages(version)
    result_list = []
    if server_list:
        result_list += [BetaTargetNameServerType(ip, version) for ip in server_list]
    if private_server_list:
        result_list += [BetaPrivateTargetNameServerType(ip, version) for ip in private_server_list]
    return messages.PolicyAlternativeNameServerConfig(targetNameServers=result_list)