from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.data_catalog import crawlers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
def _GetBucketsPatch(args, crawler, messages):
    """Returns list of buckets for a patch request based on the args provided.

  Args:
    args: The parsed args namespace.
    crawler: CachedResult, The cached crawler result.
    messages: The messages module.
  Returns:
    Desired list of buckets.
  """
    bucket_scope = crawler.Get().config.bucketScope
    buckets = bucket_scope.buckets if bucket_scope else []
    if args.IsSpecified('clear_buckets'):
        buckets = []
    if args.IsSpecified('remove_buckets'):
        to_remove = set(args.remove_buckets)
        buckets = [b for b in buckets if b.bucket not in to_remove]
    if args.IsSpecified('add_buckets'):
        buckets += [messages.GoogleCloudDatacatalogV1alpha3BucketSpec(bucket=b) for b in args.add_buckets]
    return buckets