from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
def CreateAliasIpRangeMessagesFromString(messages, instance, alias_ip_ranges_string):
    """Returns a list of AliasIpRange messages by parsing the input string.

  Args:
    messages: GCE API messages.
    instance: If True, this call is for parsing instance flags; otherwise
        it is for instance template.
    alias_ip_ranges_string: Command line string that specifies a list of
        alias IP ranges. Alias IP ranges are separated by semicolons.
        Each alias IP range has the format <alias-ip-range> or
        {range-name}:<alias-ip-range>.  The range-name is the name of the
        range within the network interface's subnet from which to allocate
        an alias range. alias-ip-range can be a CIDR range, an IP address,
        or a net mask (e.g. "/24"). Note that the validation is done on
        the server. This method just creates the request message by parsing
        the input string.
        Example string:
        "/24;range2:192.168.100.0/24;range3:192.168.101.0/24"

  Returns:
    A list of AliasIpRange messages.
  """
    if not alias_ip_ranges_string:
        return []
    alias_ip_range_strings = alias_ip_ranges_string.split(';')
    return [_CreateAliasIpRangeMessageFromString(messages, instance, s) for s in alias_ip_range_strings]