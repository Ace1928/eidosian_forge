from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddPlacementTypeFlag(parser, for_node_pool=False, hidden=False):
    """Adds a --placement-type flag to parser."""
    if for_node_pool:
        help_text = textwrap.dedent('      Placement type allows to define the type of node placement within this node\n      pool.\n\n      `UNSPECIFIED` - No requirements on the placement of nodes. This is the\n      default option.\n\n      `COMPACT` - GKE will attempt to place the nodes in a close proximity to each\n      other. This helps to reduce the communication latency between the nodes, but\n      imposes additional limitations on the node pool size.\n\n        $ {command} node-pool-1 --cluster=example-cluster --placement-type=COMPACT\n      ')
    else:
        help_text = textwrap.dedent('      Placement type allows to define the type of node placement within the default\n      node pool of this cluster.\n\n      `UNSPECIFIED` - No requirements on the placement of nodes. This is the\n      default option.\n\n      `COMPACT` - GKE will attempt to place the nodes in a close proximity to each\n      other. This helps to reduce the communication latency between the nodes, but\n      imposes additional limitations on the node pool size.\n\n        $ {command} example-cluster --placement-type=COMPACT\n      ')
    parser.add_argument('--placement-type', choices=api_adapter.PLACEMENT_OPTIONS, help=help_text, hidden=hidden)