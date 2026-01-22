from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddPriority(parser, operation, is_plural=False):
    """Adds the priority argument to the argparse."""
    parser.add_argument('name' + ('s' if is_plural else ''), metavar='PRIORITY', nargs='*' if is_plural else None, completer=SecurityPolicyRulesCompleter, help='The priority of the rule{0} to {1}. Rules are evaluated in order from highest priority to lowest priority where 0 is the highest priority and 2147483647 is the lowest priority.'.format('s' if is_plural else '', operation))