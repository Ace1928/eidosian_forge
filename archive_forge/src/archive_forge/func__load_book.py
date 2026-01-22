import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
@tenacity.retry(retry=tenacity.retry_if_exception_type(exception_types=excp.StorageFailure), stop=tenacity.stop_after_attempt(RETRY_ATTEMPTS), wait=tenacity.wait_fixed(RETRY_WAIT_TIMEOUT), reraise=True)
def _load_book(self):
    book_uuid = self.book_uuid
    if self._backend is not None and book_uuid is not None:
        with contextlib.closing(self._backend.get_connection()) as conn:
            return conn.get_logbook(book_uuid)
    return None