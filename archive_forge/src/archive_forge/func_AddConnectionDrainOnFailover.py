from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.network_services import completers as network_services_completers
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddConnectionDrainOnFailover(parser, default):
    """Adds the connection drain on failover argument to the argparse."""
    parser.add_argument('--connection-drain-on-failover', action='store_true', default=default, help='      Applicable only for backend service-based external and internal\n      passthrough Network Load Balancers as part of a connection tracking\n      policy. Only applicable when the backend service protocol is TCP. Not\n      applicable to any other load balancer. Enabled by default, this option\n      instructs the load balancer to allow established TCP connections to\n      persist for up to 300 seconds on instances or endpoints in primary\n      backends during failover, and on instances or endpoints in failover\n      backends during failback. For details, see: [Connection draining on\n      failover and failback for internal passthrough Network Load\n      Balancers](https://cloud.google.com/load-balancing/docs/internal/failover-overview#connection_draining)\n      and [Connection draining on failover and failback for external passthrough\n      Network Load\n      Balancers](https://cloud.google.com/load-balancing/docs/network/networklb-failover-overview#connection_draining).\n      ')