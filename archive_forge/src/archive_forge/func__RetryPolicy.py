from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _RetryPolicy(self, min_retry_delay, max_retry_delay):
    """Builds RetryPolicy message from argument values.

    Args:
      min_retry_delay (str): The minimum delay between consecutive deliveries of
        a given message.
      max_retry_delay (str): The maximum delay between consecutive deliveries of
        a given message.

    Returns:
      DeadLetterPolicy message or None.
    """
    if min_retry_delay or max_retry_delay:
        return self.messages.RetryPolicy(minimumBackoff=min_retry_delay, maximumBackoff=max_retry_delay)
    return None