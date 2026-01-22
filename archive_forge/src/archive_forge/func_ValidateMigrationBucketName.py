from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateMigrationBucketName(bucket_name):
    """Validates the Cloud Storage bucket name used for CDC during migration, should not start with 'gs://'.

  Args:
    bucket_name: the Cloud Storage bucket name.

  Returns:
    the Cloud Storage bucket name.
  Raises:
    BadArgumentException: when the Cloud Storage bucket name doesn't conform to
    the pattern.
  """
    pattern = re.compile('^(?!gs://)([a-z0-9\\._-]+)$')
    if not pattern.match(bucket_name):
        raise exceptions.BadArgumentException('--bucket', 'Invalid bucket name')
    return bucket_name