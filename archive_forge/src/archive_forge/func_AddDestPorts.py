from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddDestPorts(parser, required=False):
    """Adds the destination ports."""
    parser.add_argument('--dest-ports', type=arg_parsers.ArgList(), required=required, metavar='DEST_PORTS', help='A list of destination protocols and ports to which the firewall rule will apply.')