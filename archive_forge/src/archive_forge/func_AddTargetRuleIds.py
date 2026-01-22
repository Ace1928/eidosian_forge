from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddTargetRuleIds(parser, is_add):
    """Adds target-rule-ids argument to the argparse."""
    parser.add_argument('--target-rule-ids', type=arg_parsers.ArgList(), metavar='RULE_ID', help=_WAF_EXCLUSION_TARGET_RULE_IDS_HELP_TEXT_FOR_ADD if is_add else _WAF_EXCLUSION_TARGET_RULE_IDS_HELP_TEXT_FOR_REMOVE)