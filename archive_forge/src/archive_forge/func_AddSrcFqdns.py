from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddSrcFqdns(parser):
    """Adds source fqdns to this rule."""
    parser.add_argument('--src-fqdns', type=arg_parsers.ArgList(), metavar='SOURCE_FQDNS', required=False, help='Source FQDNs to match for this rule. Can only be specified if DIRECTION is `ingress`.')