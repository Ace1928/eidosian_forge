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
def AddCacheKeyQueryStringList(parser):
    """Adds cache key include/exclude query string flags to the argparse."""
    cache_key_query_string_list = parser.add_mutually_exclusive_group()
    cache_key_query_string_list.add_argument('--cache-key-query-string-whitelist', type=arg_parsers.ArgList(min_length=1), metavar='QUERY_STRING', default=None, help="      Specifies a comma separated list of query string parameters to include\n      in cache keys. All other parameters will be excluded. Either specify\n      --cache-key-query-string-whitelist or --cache-key-query-string-blacklist,\n      not both. '&' and '=' will be percent encoded and not treated as\n      delimiters. Can only be applied for global resources.\n      ")
    cache_key_query_string_list.add_argument('--cache-key-query-string-blacklist', type=arg_parsers.ArgList(), metavar='QUERY_STRING', default=None, help="      Specifies a comma separated list of query string parameters to exclude\n      in cache keys. All other parameters will be included. Either specify\n      --cache-key-query-string-whitelist or --cache-key-query-string-blacklist,\n      not both. '&' and '=' will be percent encoded and not treated as\n      delimiters. Can only be applied for global resources.\n      ")