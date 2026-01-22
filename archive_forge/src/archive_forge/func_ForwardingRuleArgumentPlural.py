from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.addresses import flags as addresses_flags
from googlecloudsdk.command_lib.util import completers
def ForwardingRuleArgumentPlural(required=True):
    return compute_flags.ResourceArgument(resource_name='forwarding rule', completer=ForwardingRulesCompleter, plural=True, required=required, regional_collection='compute.forwardingRules', global_collection='compute.globalForwardingRules', region_explanation=compute_flags.REGION_PROPERTY_EXPLANATION)