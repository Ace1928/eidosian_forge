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
def AddServiceLoadBalancingPolicy(parser, required=False, is_update=False):
    """Add support for --service-lb-policy flag."""
    group = parser.add_mutually_exclusive_group() if is_update else parser
    group.add_argument('--service-lb-policy', metavar='SERVICE_LOAD_BALANCING_POLICY', required=required, help=SERVICE_LB_POLICY_HELP)
    if is_update:
        group.add_argument('--no-service-lb-policy', required=False, action='store_true', default=None, help='No service load balancing policies should be attached to the backend service.')