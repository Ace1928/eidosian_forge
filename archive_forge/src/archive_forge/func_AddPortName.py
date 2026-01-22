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
def AddPortName(parser):
    """Add port-name flag."""
    parser.add_argument('--port-name', help="      Backend services for Application Load Balancers and proxy Network\n      Load Balancers must reference exactly one named port if using instance\n      group backends.\n\n      Each instance group backend exports one or more named ports, which map a\n      user-configurable name to a port number. The backend service's named port\n      subscribes to one named port on each instance group. The resolved port\n      number can differ among instance group backends, based on each instance\n      group's named port list.\n\n      When omitted, a backend service subscribes to a named port called http.\n\n      The named port for a backend service is either ignored or cannot be set\n      for these load balancing configurations:\n\n      - For any load balancer, if the backends are not instance groups\n        (for example, GCE_VM_IP_PORT NEGs).\n      - For any type of backend on a backend service for internal or external\n        passthrough Network Load Balancers.\n\n      See also\n      https://cloud.google.com/load-balancing/docs/backend-service#named_ports.\n      ")