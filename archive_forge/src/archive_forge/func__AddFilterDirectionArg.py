from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddFilterDirectionArg(parser):
    """Adds args to specify filter direction."""
    parser.add_argument('--filter-direction', choices=['both', 'egress', 'ingress'], metavar='DIRECTION', help='        - For `ingress`, only ingress traffic is mirrored.\n        - For `egress`, only egress traffic is mirrored.\n        - For `both` (default), both directions are mirrored.\n        ')