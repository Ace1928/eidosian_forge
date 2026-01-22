from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def ParseRules(rules, message_classes, action=ActionType.ALLOW):
    """Parses protocol:port mappings from --allow or --rules command line."""
    rule_value_list = []
    for spec in rules or []:
        match = LEGAL_SPECS.match(spec)
        if not match:
            raise compute_exceptions.ArgumentError('Firewall rules must be of the form {0}; received [{1}].'.format(ALLOWED_METAVAR, spec))
        if match.group('ports'):
            ports = [match.group('ports')]
        else:
            ports = []
        if action == ActionType.ALLOW:
            rule = message_classes.Firewall.AllowedValueListEntry(IPProtocol=match.group('protocol'), ports=ports)
        else:
            rule = message_classes.Firewall.DeniedValueListEntry(IPProtocol=match.group('protocol'), ports=ports)
        rule_value_list.append(rule)
    return rule_value_list