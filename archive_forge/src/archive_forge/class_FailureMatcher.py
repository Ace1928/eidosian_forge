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
class FailureMatcher(object):
    """Needed for failure objects comparison."""

    def __init__(self, failure):
        self._failure = failure

    def __repr__(self):
        return str(self._failure)

    def __eq__(self, other):
        return self._failure.matches(other)

    def __ne__(self, other):
        return not self.__eq__(other)