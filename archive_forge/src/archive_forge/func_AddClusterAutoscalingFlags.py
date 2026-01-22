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
def AddClusterAutoscalingFlags(parser, update_group=None, hidden=False):
    """Adds autoscaling related flags to parser.

  Autoscaling related flags are: --enable-autoscaling
  --min-nodes --max-nodes --location-policy flags.

  Args:
    parser: A given parser.
    update_group: An optional group of mutually exclusive flag options to which
      an --enable-autoscaling flag is added.
    hidden: If true, suppress help text for added options.

  Returns:
    Argument group for autoscaling flags.
  """
    group = parser.add_argument_group('Cluster autoscaling')
    autoscaling_group = group if update_group is None else update_group
    autoscaling_group.add_argument('--enable-autoscaling', default=None, help='Enables autoscaling for a node pool.\n\nEnables autoscaling in the node pool specified by --node-pool or\nthe default node pool if --node-pool is not provided. If not already,\n--max-nodes or --total-max-nodes must also be set.', hidden=hidden, action='store_true')
    group.add_argument('--max-nodes', help='Maximum number of nodes per zone in the node pool.\n\nMaximum number of nodes per zone to which the node pool specified by --node-pool\n(or default node pool if unspecified) can scale. Ignored unless\n--enable-autoscaling is also specified.', hidden=hidden, type=int)
    group.add_argument('--min-nodes', help='Minimum number of nodes per zone in the node pool.\n\nMinimum number of nodes per zone to which the node pool specified by --node-pool\n(or default node pool if unspecified) can scale. Ignored unless\n--enable-autoscaling is also specified.', hidden=hidden, type=int)
    group.add_argument('--total-max-nodes', help='Maximum number of all nodes in the node pool.\n\nMaximum number of all nodes to which the node pool specified by --node-pool\n(or default node pool if unspecified) can scale. Ignored unless\n--enable-autoscaling is also specified.', hidden=hidden, type=int)
    group.add_argument('--total-min-nodes', help='Minimum number of all nodes in the node pool.\n\nMinimum number of all nodes to which the node pool specified by --node-pool\n(or default node pool if unspecified) can scale. Ignored unless\n--enable-autoscaling is also specified.', hidden=hidden, type=int)
    group.add_argument('--location-policy', choices=api_adapter.LOCATION_POLICY_OPTIONS, help='Location policy specifies the algorithm used when scaling-up the node pool.\n\n* `BALANCED` - Is a best effort policy that aims to balance the sizes of available\n  zones.\n* `ANY` - Instructs the cluster autoscaler to prioritize utilization of unused\n  reservations, and reduces preemption risk for Spot VMs.', hidden=hidden)
    return group