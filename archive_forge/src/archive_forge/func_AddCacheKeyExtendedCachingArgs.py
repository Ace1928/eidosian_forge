from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCacheKeyExtendedCachingArgs(parser):
    """Adds cache key includeHttpHeader and includeNamedCookie flags to the argparse."""
    parser.add_argument('--cache-key-include-http-header', type=arg_parsers.ArgList(), metavar='HEADER_FIELD_NAME', help='      Specifies a comma-separated list of HTTP headers, by field name, to\n      include in cache keys. Only the request URL is included in the cache\n      key by default.\n      ')
    parser.add_argument('--cache-key-query-string-whitelist', type=arg_parsers.ArgList(), metavar='QUERY_STRING', help="      Specifies a comma-separated list of query string parameters to include\n      in cache keys. Default parameters are always included. '&' and '=' are\n      percent encoded and not treated as delimiters.\n      ")