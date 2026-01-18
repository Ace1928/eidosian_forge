import os
import uuid
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
Returns the default regional bucket name.

  Args:
    region: Cloud Run region.

  Returns:
    GCS bucket name.
  