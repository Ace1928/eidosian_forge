from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddPerformMaintenanceNodesArgToParser(parser):
    """Add --nodes flag."""
    parser.add_argument('--nodes', required=True, metavar='NODE', type=arg_parsers.ArgList(min_length=1), help='The names of the nodes to perform maintenance on.')