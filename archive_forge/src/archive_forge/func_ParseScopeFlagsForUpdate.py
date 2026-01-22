from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ParseScopeFlagsForUpdate(ref, args, request, crawler):
    """Python hook that parses the crawl scope args into the update request.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The update crawler request.
    crawler: CachedResult, The cached crawler result.
  Returns:
    Request with crawl scope set appropriately.
  """
    del ref
    client = crawlers.CrawlersClient()
    messages = client.messages
    if _IsChangeBucketsSpecified(args):
        buckets = _GetBucketsPatch(args, crawler, messages)
        crawl_scope = 'bucket'
    else:
        buckets = None
        crawl_scope = args.crawl_scope
    return _SetScopeInRequest(crawl_scope, buckets, request, messages)