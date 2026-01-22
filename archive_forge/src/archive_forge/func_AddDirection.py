from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddDirection(parser, required=False):
    """Adds the direction of the traffic to which the rule is applied."""
    parser.add_argument('--direction', required=required, choices=['INGRESS', 'EGRESS'], help='Direction of the traffic the rule is applied. The default is to apply on incoming traffic.')