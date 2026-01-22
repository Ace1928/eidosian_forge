from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
def GetNegativeCachingPolicy(client, args, backend_bucket):
    """Returns the negative caching policy.

  Args:
    client: The client used by gcloud.
    args: The arguments passed to the gcloud command.
    backend_bucket: The backend bucket object. If the backend bucket object
      contains a negative caching policy already, it is used as the base to
      apply changes based on args.

  Returns:
    The negative caching policy.
  """
    negative_caching_policy = None
    if args.negative_caching_policy:
        negative_caching_policy = []
        for code, ttl in args.negative_caching_policy.items():
            negative_caching_policy.append(client.messages.BackendBucketCdnPolicyNegativeCachingPolicy(code=code, ttl=ttl))
    elif backend_bucket.cdnPolicy is not None and backend_bucket.cdnPolicy.negativeCachingPolicy is not None:
        negative_caching_policy = backend_bucket.cdnPolicy.negativeCachingPolicy
    return negative_caching_policy