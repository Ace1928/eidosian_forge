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
def AddCacheKeyIncludeQueryString(parser, default):
    """Adds cache key include/exclude query string flag to the argparse."""
    update_command = default is None
    if update_command:
        update_command_help = '        Enable including query string in cache key. If enabled, the query string\n        parameters will be included according to\n        --cache-key-query-string-whitelist and\n        --cache-key-query-string-blacklist. If disabled, the entire query string\n        will be excluded. Use "--cache-key-query-string-blacklist=" (sets the\n        blacklist to the empty list) to include the entire query string. Can\n        only be applied for global resources.\n        '
    else:
        update_command_help = '        Enable including query string in cache key. If enabled, the query string\n        parameters will be included according to\n        --cache-key-query-string-whitelist and\n        --cache-key-query-string-blacklist. If neither is set, the entire query\n        string will be included. If disabled, then the entire query string will\n        be excluded. Can only be applied for global resources.\n        '
    parser.add_argument('--cache-key-include-query-string', action='store_true', default=default, help=update_command_help)