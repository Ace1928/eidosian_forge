from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.core.console import console_io
import six
def _CreateBucketIfNotExists(bucket):
    """Creates a Cloud Storage bucket if it doesn't exist."""
    if storage_helpers.GetBucket(bucket):
        return
    region = console_io.PromptResponse(message='The bucket [{}] doesn\'t exist. Please enter a Cloud Storage region to create the bucket (leave empty to create in "global" region):'.format(bucket))
    storage_helpers.CreateBucketIfNotExists(bucket, region)