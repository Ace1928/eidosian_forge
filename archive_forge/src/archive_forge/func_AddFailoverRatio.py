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
def AddFailoverRatio(parser):
    """Adds the failover ratio argument to the argparse."""
    parser.add_argument('--failover-ratio', type=arg_parsers.BoundedFloat(lower_bound=0.0, upper_bound=1.0), help='      Applicable only to backend service-based external passthrough Network load\n      balancers and internal passthrough Network load balancers as part of a\n      failover policy. Not applicable to any other load balancer. This option\n      defines the ratio used to control when failover and failback occur.\n      For details, see: [Failover ratio for internal passthrough Network\n      Load Balancers](https://cloud.google.com/load-balancing/docs/internal/failover-overview#failover_ratio)\n      and [Failover ratio for external passthrough Network Load Balancer\n      overview](https://cloud.google.com/load-balancing/docs/network/networklb-failover-overview#failover_ratio).\n      ')