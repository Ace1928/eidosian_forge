import abc
import concurrent.futures
from google.api_core import exceptions
from google.api_core import retry as retries
from google.api_core.future import _helpers
from google.api_core.future import base
def _blocking_poll(self, timeout=_DEFAULT_VALUE, retry=None, polling=None):
    """Poll and wait for the Future to be resolved."""
    if self._result_set:
        return
    polling = polling or self._polling
    if timeout is not PollingFuture._DEFAULT_VALUE:
        polling = polling.with_timeout(timeout)
    try:
        polling(self._done_or_raise)(retry=retry)
    except exceptions.RetryError:
        raise concurrent.futures.TimeoutError(f'Operation did not complete within the designated timeout of {polling.timeout} seconds.')