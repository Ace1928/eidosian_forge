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
def AddConnectionDrainingTimeout(parser):
    parser.add_argument('--connection-draining-timeout', type=arg_parsers.Duration(upper_bound='1h'), help='      Connection draining timeout to be used during removal of VMs from\n      instance groups. This guarantees that for the specified time all existing\n      connections to a VM will remain untouched, but no new connections will be\n      accepted. Set timeout to zero to disable connection draining. Enable\n      feature by specifying a timeout of up to one hour.\n      If the flag is omitted API default value (0s) will be used.\n      See $ gcloud topic datetimes for information on duration formats.\n      ')