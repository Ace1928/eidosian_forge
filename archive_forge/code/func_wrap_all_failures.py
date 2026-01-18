import contextlib
import string
import threading
import time
from oslo_utils import timeutils
import redis
from taskflow import exceptions
from taskflow.listeners import capturing
from taskflow.persistence.backends import impl_memory
from taskflow import retry
from taskflow import task
from taskflow.types import failure
from taskflow.utils import kazoo_utils
from taskflow.utils import redis_utils
@contextlib.contextmanager
def wrap_all_failures():
    """Convert any exceptions to WrappedFailure.

    When you expect several failures, it may be convenient
    to wrap any exception with WrappedFailure in order to
    unify error handling.
    """
    try:
        yield
    except Exception:
        raise exceptions.WrappedFailure([failure.Failure()])