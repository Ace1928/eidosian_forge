from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def FindBucket(self, project, prefix):
    """Gets the first bucket the project has access to with a matching prefix.

    Args:
      project: The id string of the project the bucket is associated with.
      prefix: The string literal prefix of the bucket being searched for.

    Returns:
      The first bucket message object found matching the prefix, or none.
    """
    messages = self._storage_client.MESSAGES_MODULE
    response = self._storage_client.buckets.List(messages.StorageBucketsListRequest(prefix=prefix, project=project))
    for bucket in response.items:
        return bucket
    return None