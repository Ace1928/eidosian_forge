from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as forwarding_rule_flags
def AddTargetServiceAndProducerForwardingRuleArgs(parser):
    target = parser.add_mutually_exclusive_group(required=True)
    forwarding_rule_flags.ForwardingRuleArgumentForServiceAttachment().AddArgument(parser, mutex_group=target)
    target.add_argument('--target-service', required=False, help='URL of the target service that receives forwarded traffic.')