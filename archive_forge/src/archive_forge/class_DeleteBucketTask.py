from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
class DeleteBucketTask(CloudDeleteTask):
    """Task to delete a bucket."""

    def _make_delete_api_call(self, client, request_config):
        try:
            client.delete_bucket(self._url.bucket_name, request_config)
        except Exception as error:
            if 'not empty' in str(error):
                raise type(error)('Bucket is not empty. To delete all objects and then delete bucket, use: gcloud storage rm -r')
            else:
                raise