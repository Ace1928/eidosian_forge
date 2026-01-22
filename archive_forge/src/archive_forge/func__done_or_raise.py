import abc
import concurrent.futures
from google.api_core import exceptions
from google.api_core import retry as retries
from google.api_core.future import _helpers
from google.api_core.future import base
def _done_or_raise(self, retry=None):
    """Check if the future is done and raise if it's not."""
    if not self.done(retry=retry):
        raise _OperationNotComplete()