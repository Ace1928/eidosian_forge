from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import hashlib
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.storage import storage_parallel
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import times
from googlecloudsdk.third_party.appengine.tools import context_util
from six.moves import map  # pylint: disable=redefined-builtin
def _GetLifecycleDeletePolicy(storage_client, bucket_ref):
    """Get the TTL of objects in days as specified by the lifecycle policy.

  Only "delete by age" policies are accounted for.

  Args:
    storage_client: storage_api.StorageClient, API client wrapper.
    bucket_ref: The GCS bucket reference.

  Returns:
    datetime.timedelta, TTL of objects in days, or None if no deletion
    policy on the bucket.
  """
    try:
        bucket = storage_client.client.buckets.Get(request=storage_client.messages.StorageBucketsGetRequest(bucket=bucket_ref.bucket), global_params=storage_client.messages.StandardQueryParameters(fields='lifecycle'))
    except apitools_exceptions.HttpForbiddenError:
        return None
    if not bucket.lifecycle:
        return None
    rules = bucket.lifecycle.rule
    ages = [rule.condition.age for rule in rules if rule.condition.age is not None and rule.condition.age >= 0 and (rule.action.type == 'Delete')]
    return datetime.timedelta(min(ages)) if ages else None