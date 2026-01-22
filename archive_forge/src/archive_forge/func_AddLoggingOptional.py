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
def AddLoggingOptional(parser):
    """Adds the logging optional argument to the argparse."""
    parser.add_argument('--logging-optional', choices=['EXCLUDE_ALL_OPTIONAL', 'INCLUDE_ALL_OPTIONAL', 'CUSTOM'], type=arg_utils.ChoiceToEnumName, help='      This field can only be specified if logging is enabled for the backend\n      service. Configures whether all, none, or a subset of optional fields\n      should be added to the reported logs. Default is EXCLUDE_ALL_OPTIONAL.\n      This field can only be specified for internal and external passthrough\n      Network Load Balancers.\n      ')