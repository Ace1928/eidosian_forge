from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
def SortFirewallPolicyRules(client, rules):
    """Sort the organization firewall rules by direction and priority."""
    ingress_org_firewall_rule = [item for item in rules if item.direction == client.messages.FirewallPolicyRule.DirectionValueValuesEnum.INGRESS]
    ingress_org_firewall_rule.sort(key=lambda x: x.priority, reverse=False)
    egress_org_firewall_rule = [item for item in rules if item.direction == client.messages.FirewallPolicyRule.DirectionValueValuesEnum.EGRESS]
    egress_org_firewall_rule.sort(key=lambda x: x.priority, reverse=False)
    return ingress_org_firewall_rule + egress_org_firewall_rule