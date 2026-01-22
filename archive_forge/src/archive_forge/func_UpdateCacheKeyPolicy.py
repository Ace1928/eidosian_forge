from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def UpdateCacheKeyPolicy(args, cache_key_policy):
    """Sets the cache_key_policy according to the command line arguments.

  Args:
    args: Arguments specified through command line.
    cache_key_policy: new CacheKeyPolicy to be set (or preexisting one if using
      update).
  """
    if args.cache_key_include_protocol is not None:
        cache_key_policy.includeProtocol = args.cache_key_include_protocol
    if args.cache_key_include_host is not None:
        cache_key_policy.includeHost = args.cache_key_include_host
    if args.cache_key_include_query_string is not None:
        cache_key_policy.includeQueryString = args.cache_key_include_query_string
        if not args.cache_key_include_query_string:
            cache_key_policy.queryStringWhitelist = []
            cache_key_policy.queryStringBlacklist = []
    if args.cache_key_query_string_whitelist is not None:
        cache_key_policy.queryStringWhitelist = args.cache_key_query_string_whitelist
        cache_key_policy.includeQueryString = True
        cache_key_policy.queryStringBlacklist = []
    if args.cache_key_query_string_blacklist is not None:
        cache_key_policy.queryStringBlacklist = args.cache_key_query_string_blacklist
        cache_key_policy.includeQueryString = True
        cache_key_policy.queryStringWhitelist = []
    if args.cache_key_include_http_header is not None:
        cache_key_policy.includeHttpHeaders = args.cache_key_include_http_header
    if args.cache_key_include_named_cookie is not None:
        cache_key_policy.includeNamedCookies = args.cache_key_include_named_cookie