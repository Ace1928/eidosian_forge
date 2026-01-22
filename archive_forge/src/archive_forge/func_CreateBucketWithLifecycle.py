from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
def CreateBucketWithLifecycle(self, days_to_live=10):
    """Creates a bucket object that deletes its contents after a number of days.

    Args:
      days_to_live: The number of days to wait before deleting an item within
        this bucket. Count starts when the item is created.

    Returns:
      A bucket message object that has not yet been created in Cloud Storage.
    """
    messages = self._storage_client.MESSAGES_MODULE
    lifecycle = messages.Bucket.LifecycleValue()
    lifecycle_rule = messages.Bucket.LifecycleValue.RuleValueListEntry()
    lifecycle_rule.condition = messages.Bucket.LifecycleValue.RuleValueListEntry.ConditionValue()
    lifecycle_rule.condition.age = days_to_live
    lifecycle_rule.action = messages.Bucket.LifecycleValue.RuleValueListEntry.ActionValue(type='Delete')
    lifecycle.rule.append(lifecycle_rule)
    return messages.Bucket(lifecycle=lifecycle)