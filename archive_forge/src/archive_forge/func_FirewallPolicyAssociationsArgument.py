from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def FirewallPolicyAssociationsArgument(required=False, plural=False):
    return compute_flags.ResourceArgument(name='name', resource_name='association', completer=FirewallPoliciesCompleter, plural=plural, required=required, global_collection='compute.firewallPolicies')