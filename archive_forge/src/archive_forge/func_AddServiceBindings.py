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
def AddServiceBindings(parser, required=False, is_update=False, help_text=None):
    """Add support for --service_bindings flag."""
    group = parser.add_mutually_exclusive_group() if is_update else parser
    group.add_argument('--service-bindings', metavar='SERVICE_BINDING', required=required, type=arg_parsers.ArgList(min_length=1), completer=network_services_completers.ServiceBindingsCompleter, help=help_text if help_text is not None else SERVICE_BINDINGS_HELP)
    if is_update:
        group.add_argument('--no-service-bindings', required=False, action='store_true', default=None, help='No service bindings should be attached to the backend service.')