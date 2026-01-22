from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _confirm_and_lock_retention_policy(self, api_client, bucket_resource, request_config):
    """Locks a buckets retention policy if possible and the user confirms.

    Args:
      api_client (cloud_api.CloudApi): API client that should issue the lock
        request.
      bucket_resource (BucketResource): Metadata of the bucket containing the
        retention policy to lock.
      request_config (request_config_factory._RequestConfig): Contains
        additional request parameters.
    """
    lock_prompt = 'This will permanently set the retention policy on "{}" to the following:\n\n{}\n\nThis setting cannot be reverted. Continue? '.format(self._bucket_resource, bucket_resource.retention_policy)
    if not bucket_resource.retention_policy:
        raise command_errors.Error('Bucket "{}" does not have a retention policy.'.format(self._bucket_resource))
    elif bucket_resource.retention_policy_is_locked:
        log.error('Retention policy on "{}" is already locked.'.format(self._bucket_resource))
    elif console_io.PromptContinue(message=lock_prompt, default=False):
        log.status.Print('Locking retention policy on {}...'.format(self._bucket_resource))
        api_client.lock_bucket_retention_policy(bucket_resource, request_config)
    else:
        log.error('Abort locking retention policy on "{}".'.format(self._bucket_resource))