from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
def ParseInterfaces(interfaces, message_classes):
    """Parses id=ip_address mappings from --interfaces command line."""
    if len(interfaces) != 1 and len(interfaces) != 2 and (len(interfaces) != 4):
        raise exceptions.ArgumentError('Number of interfaces must be either one, two, or four; received [{0}] interface(s).'.format(len(interfaces)))
    interface_list = []
    for spec in interfaces or []:
        match_ipv4 = LEGAL_SPECS.match(spec)
        if match_ipv4:
            interface_id = match_ipv4.group('id')
            ip_address = match_ipv4.group('ipAddress')
            interface = message_classes.ExternalVpnGatewayInterface(id=int(interface_id), ipAddress=ip_address)
            interface_list.append(interface)
            continue
        match_ipv6 = LEGAL_IPV6_SPECS.match(spec)
        if match_ipv6:
            interface_id = match_ipv6.group('id')
            ipv6_address = match_ipv6.group('ipv6Address')
            interface = message_classes.ExternalVpnGatewayInterface(id=int(interface_id), ipv6Address=ipv6_address)
            interface_list.append(interface)
            continue
        if not match_ipv4 and (not match_ipv6):
            raise exceptions.ArgumentError('Interfaces must be of the form {0}, ID must be an integer value in [0,1,2,3], IP_ADDRESS must be a valid IP address; received [{1}].'.format(ALLOWED_METAVAR, spec))
    return interface_list