from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddLayer4Configs(parser, required=False):
    """Adds the layer4 configs."""
    parser.add_argument('--layer4-configs', type=arg_parsers.ArgList(), required=required, metavar='LAYER4_CONFIG', help='A list of destination protocols and ports to which the firewall rule will apply.')