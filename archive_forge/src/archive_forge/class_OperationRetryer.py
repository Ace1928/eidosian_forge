from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker as tracker
from googlecloudsdk.core.util import retry
class OperationRetryer(object):
    """A wrapper around a Retryer that works with CRM operations.

  Uses predefined constants for retry timing, so all CRM operation commands can
  share their retry timing settings.
  """

    def __init__(self, pre_start_sleep=lambda: time.sleep(1), max_retry_ms=2000, max_wait_ms=300000, wait_ceiling_ms=20000, first_retry_sleep_ms=2000):
        self._pre_start_sleep = pre_start_sleep
        self._max_retry_ms = max_retry_ms
        self._max_wait_ms = max_wait_ms
        self._wait_ceiling_ms = wait_ceiling_ms
        self._first_retry_sleep_ms = first_retry_sleep_ms

    def RetryPollOperation(self, operation_poller, operation):
        self._pre_start_sleep()
        return self._Retryer().RetryOnResult(lambda: operation_poller.Poll(operation), should_retry_if=self._ShouldRetry, sleep_ms=self._first_retry_sleep_ms)

    def _Retryer(self):
        return retry.Retryer(exponential_sleep_multiplier=2, max_wait_ms=self._max_wait_ms, wait_ceiling_ms=self._wait_ceiling_ms)

    def _ShouldRetry(self, result, state):
        if isinstance(result, exceptions.HttpError):
            return self._CheckTimePassedBelowMax(result, state)
        return self._CheckResultNotException(result)

    def _CheckTimePassedBelowMax(self, result, state):
        if state.time_passed_ms > self._max_retry_ms:
            raise result
        return True

    def _CheckResultNotException(self, result):
        if isinstance(result, Exception):
            raise result
        return not result.done