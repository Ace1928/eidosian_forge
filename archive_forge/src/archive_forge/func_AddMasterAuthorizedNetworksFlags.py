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
def AddMasterAuthorizedNetworksFlags(parser, enable_group_for_update=None):
    """Adds Master Authorized Networks related flags to parser.

  Master Authorized Networks related flags are:
  --enable-master-authorized-networks --master-authorized-networks.

  Args:
    parser: A given parser.
    enable_group_for_update: An optional group of mutually exclusive flag
      options to which an --enable-master-authorized-networks flag is added in
      an update command.
  """
    if enable_group_for_update is None:
        master_flag_group = parser.add_argument_group('Master Authorized Networks')
        enable_flag_group = master_flag_group
    else:
        master_flag_group = parser.add_argument_group('')
        enable_flag_group = enable_group_for_update
    enable_flag_group.add_argument('--enable-master-authorized-networks', default=None, help='Allow only specified set of CIDR blocks (specified by the\n`--master-authorized-networks` flag) to connect to Kubernetes master through\nHTTPS. Besides these blocks, the following have access as well:\n\n  1) The private network the cluster connects to if\n  `--enable-private-nodes` is specified.\n  2) Google Compute Engine Public IPs if `--enable-private-nodes` is not\n  specified.\n\nUse `--no-enable-master-authorized-networks` to disable. When disabled, public\ninternet (0.0.0.0/0) is allowed to connect to Kubernetes master through HTTPS.\n', action='store_true')
    master_flag_group.add_argument('--master-authorized-networks', type=arg_parsers.ArgList(min_length=1), metavar='NETWORK', help='The list of CIDR blocks (up to {max_private} for private cluster, {max_public} for public cluster) that are allowed to connect to Kubernetes master through HTTPS. Specified in CIDR notation (e.g. 1.2.3.4/30). Cannot be specified unless `--enable-master-authorized-networks` is also specified.'.format(max_private=api_adapter.MAX_AUTHORIZED_NETWORKS_CIDRS_PRIVATE, max_public=api_adapter.MAX_AUTHORIZED_NETWORKS_CIDRS_PUBLIC))