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
Locks a buckets retention policy if possible and the user confirms.

    Args:
      api_client (cloud_api.CloudApi): API client that should issue the lock
        request.
      bucket_resource (BucketResource): Metadata of the bucket containing the
        retention policy to lock.
      request_config (request_config_factory._RequestConfig): Contains
        additional request parameters.
    