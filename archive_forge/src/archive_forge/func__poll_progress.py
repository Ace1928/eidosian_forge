from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.transfer import jobs_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import scaled_integer
def _poll_progress(name):
    """Prints progress of operation and blocks until transfer is complete.

  Args:
    name (str|None): Poll this operation name.

  Returns:
    Apitools Operation object containing the completed operation's metadata.
  """
    complete_operation = retry.Retryer(jitter_ms=0, status_update_func=_print_progress).RetryOnResult(api_get, args=[name], should_retry_if=_is_operation_in_progress, sleep_ms=1000)
    _print_progress(complete_operation, retry.RetryerState(retrial=_LAST_RETRIAL, time_passed_ms=None, time_to_wait_ms=None))
    return complete_operation