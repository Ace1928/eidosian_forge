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
def AddAdditionalPodNetworkFlag(parser):
    """Adds --additional-pod-network flag to the given parser.

  Args:
    parser: A given parser.
  """
    spec = {'subnetwork': str, 'pod-ipv4-range': str, 'max-pods-per-node': int}
    parser.add_argument('--additional-pod-network', type=arg_parsers.ArgDict(spec=spec, required_keys=['pod-ipv4-range'], max_length=len(spec)), metavar='subnetwork=SUBNETWORK_NAME,pod-ipv4-range=SECONDARY_RANGE_NAME,[max-pods-per-node=NUM_PODS]', action='append', help='      Specify the details of a secondary range to be used for an additional pod network.\n      Not needed if you use "host" typed NIC from this network.\n      This parameter can be specified up to 35 times.\n\n      e.g. --additional-pod-network subnetwork=subnet-dp,pod-ipv4-range=sec-range-blue,max-pods-per-node=8.\n\n      *subnetwork*::: (Optional) The name of the subnetwork to link the pod network to.\n      If not specified, the pod network defaults to the subnet connected to the default network interface.\n\n      *pod-ipv4-range*::: (Required) The name of the secondary range in the subnetwork.\n      The range must hold at least (2 * MAX_PODS_PER_NODE * MAX_NODES_IN_RANGE) IPs.\n\n      *max-pods-per-node*::: (Optional) Maximum amount of pods per node that can utilize this ipv4-range.\n      Defaults to NodePool (if specified) or Cluster value.\n      ')