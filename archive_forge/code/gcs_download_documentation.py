from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.core import log
Performs downlaod.

    Args:
      retriable_in_flight (bool): Indicates if a download can be retried
        on network error, resuming the download from the last downloaded byte.

    Returns:
      The result returned by launch method.
    