from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def NetworkFirewallPolicyRuleArgument(required=False, plural=False, operation=None):
    return compute_flags.ResourceArgument(name='--firewall-policy', resource_name='firewall policy', plural=plural, required=required, short_help='Firewall policy ID with which to {0} rule.'.format(operation), global_collection='compute.networkFirewallPolicies', regional_collection='compute.regionNetworkFirewallPolicies')