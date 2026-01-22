from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def BetaPrivateTargetNameServerType(value, version='v1beta2'):
    """Build a single PrivateTargetNameServer based on 'value'.

  Args:
    value: (str) A string representation of an IPV4 ip address representing the
      PrivateTargetNameServer.
    version: (str) A string indicating the version of the API to be used, should
      be one of 'v1beta2' or 'v1alpha2'. This function will be removed after
      promoting v6 address to GA.

  Returns:
    A messages.PolicyAlternativeNameServerConfigTargetNameServer instance
    populated from the given ip address.
  """
    messages = GetMessages(version)
    if IsIPv4(value):
        return messages.PolicyAlternativeNameServerConfigTargetNameServer(ipv4Address=value, ipv6Address=None, forwardingPath=messages.PolicyAlternativeNameServerConfigTargetNameServer.ForwardingPathValueValuesEnum(1))
    else:
        return messages.PolicyAlternativeNameServerConfigTargetNameServer(ipv4Address=None, ipv6Address=value, forwardingPath=messages.PolicyAlternativeNameServerConfigTargetNameServer.ForwardingPathValueValuesEnum(1))