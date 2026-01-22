from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def ValidateScopeFlagsForUpdate(ref, args, request, crawler):
    """Validates scope flags for update.

  Args:
    ref: The crawler resource reference.
    args: The parsed args namespace.
    request: The update request.
    crawler: CachedResult, The cached crawler result.
  Returns:
    The request, if the crawl scope configuration is valid.
  Raises:
    InvalidCrawlScopeError: If the crawl scope configuration is not valid.
  """
    del ref
    change_buckets = _IsChangeBucketsSpecified(args)
    if change_buckets and args.crawl_scope != 'bucket' and (args.IsSpecified('crawl_scope') or crawler.Get().config.bucketScope is None):
        raise InvalidCrawlScopeError('Arguments `--add-buckets`, `--remove-buckets`, and `--clear-buckets` are only valid for bucket-scoped crawlers. Use `--crawl-scope=bucket` to specify a bucket-scoped crawler.')
    if not change_buckets and args.crawl_scope == 'bucket':
        raise InvalidCrawlScopeError('Must provide buckets to add or remove when updating the crawl scope of a bucket-scoped crawler.')
    return request