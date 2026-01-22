from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
def ParseDestPorts(dest_ports, message_classes):
    """Parses protocol:port mappings for --dest-ports command line."""
    dest_port_list = []
    for spec in dest_ports or []:
        match = LEGAL_SPECS.match(spec)
        if not match:
            raise exceptions.ArgumentError('Organization security policy rules must be of the form {0}; received [{1}].'.format(ALLOWED_METAVAR, spec))
        if match.group('ports'):
            ports = [match.group('ports')]
        else:
            ports = []
        dest_port = message_classes.SecurityPolicyRuleMatcherConfigDestinationPort(ipProtocol=match.group('protocol'), ports=ports)
        dest_port_list.append(dest_port)
    return dest_port_list