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
def AddCacheKeyIncludeHost(parser, default):
    """Adds cache key include/exclude host flag to the argparse."""
    parser.add_argument('--cache-key-include-host', action='store_true', default=default, help='      Enable including host in cache key. If enabled, requests to different\n      hosts will be cached separately. Can only be applied for global resources.\n      ')