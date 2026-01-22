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
def AddLoggingOptionalFields(parser):
    """Adds the logging optional argument to the argparse."""
    parser.add_argument('--logging-optional-fields', type=arg_parsers.ArgList(), metavar='LOGGING_OPTIONAL_FIELDS', help='      This field can only be specified if logging is enabled for the backend\n      service and "--logging-optional" was set to CUSTOM. Contains a\n      comma-separated list of optional fields you want to include in the logs.\n      For example: serverInstance, serverGkeDetails.cluster,\n      serverGkeDetails.pod.podNamespace. This can only be specified for\n      internal and external passthrough Network Load Balancers.\n      ')