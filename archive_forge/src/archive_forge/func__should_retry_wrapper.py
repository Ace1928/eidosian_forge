from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.core import log
def _should_retry_wrapper(self, exc_type, exc_value, exc_traceback, state):
    """Returns True if the error should be retried.

    This method also updates the start_byte to be used for request
    to be retried.

    Args:
      exc_type (type): The type of Exception.
      exc_value (Exception): The error instance.
      exc_traceback (traceback): The traceback for the exception.
      state (core.util.retry.RetryState): The state object
        maintained by the retryer.

    Returns:
      True if the error should be retried.
    """
    if not self.should_retry(exc_type, exc_value, exc_traceback):
        return False
    start_byte = self._download_stream.tell()
    if start_byte > self._start_byte:
        self._start_byte = start_byte
        state.retrial = 0
    log.debug('Retrying download from byte {} after exception: {}. Trace: {}'.format(start_byte, exc_type, exc_traceback))
    return True