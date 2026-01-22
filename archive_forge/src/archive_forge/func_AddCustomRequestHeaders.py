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
def AddCustomRequestHeaders(parser, remove_all_flag=False, default=None):
    """Adds custom request header flag to the argparse."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--custom-request-header', action='append', help='      Specifies a HTTP Header to be added by your load balancer.\n      This flag can be repeated to specify multiple headers.\n      For example:\n\n        $ {command} NAME             --custom-request-header "header-name: value"             --custom-request-header "another-header:"\n      ')
    if remove_all_flag:
        group.add_argument('--no-custom-request-headers', action='store_true', default=default, help='        Remove all custom request headers for the backend service.\n        ')