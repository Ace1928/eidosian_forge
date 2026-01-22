from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def SortNetworkFirewallRules(client, rules):
    """Sort the network firewall rules by direction and priority."""
    ingress_network_firewall = [item for item in rules if item.direction == client.messages.Firewall.DirectionValueValuesEnum.INGRESS]
    ingress_network_firewall.sort(key=lambda x: x.priority, reverse=False)
    egress_network_firewall = [item for item in rules if item.direction == client.messages.Firewall.DirectionValueValuesEnum.EGRESS]
    egress_network_firewall.sort(key=lambda x: x.priority, reverse=False)
    return ingress_network_firewall + egress_network_firewall