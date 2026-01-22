import datetime
import time
from collections import deque
from contextlib import contextmanager
from weakref import proxy
from dateutil.parser import isoparse
from kombu.utils.objects import cached_property
from vine import Thenable, barrier, promise
from . import current_app, states
from ._state import _set_task_join_will_block, task_join_will_block
from .app import app_or_default
from .exceptions import ImproperlyConfigured, IncompleteStream, TimeoutError
from .utils.graph import DependencyGraph, GraphFormatter
@Thenable.register
class EagerResult(AsyncResult):
    """Result that we know has already been executed."""

    def __init__(self, id, ret_value, state, traceback=None, name=None):
        self.id = id
        self._result = ret_value
        self._state = state
        self._traceback = traceback
        self._name = name
        self.on_ready = promise()
        self.on_ready(self)

    def then(self, callback, on_error=None, weak=False):
        return self.on_ready.then(callback, on_error)

    def _get_task_meta(self):
        return self._cache

    def __reduce__(self):
        return (self.__class__, self.__reduce_args__())

    def __reduce_args__(self):
        return (self.id, self._result, self._state, self._traceback)

    def __copy__(self):
        cls, args = self.__reduce__()
        return cls(*args)

    def ready(self):
        return True

    def get(self, timeout=None, propagate=True, disable_sync_subtasks=True, **kwargs):
        if disable_sync_subtasks:
            assert_will_not_block()
        if self.successful():
            return self.result
        elif self.state in states.PROPAGATE_STATES:
            if propagate:
                raise self.result if isinstance(self.result, Exception) else Exception(self.result)
            return self.result
    wait = get

    def forget(self):
        pass

    def revoke(self, *args, **kwargs):
        self._state = states.REVOKED

    def __repr__(self):
        return f'<EagerResult: {self.id}>'

    @property
    def _cache(self):
        return {'task_id': self.id, 'result': self._result, 'status': self._state, 'traceback': self._traceback, 'name': self._name}

    @property
    def result(self):
        """The tasks return value."""
        return self._result

    @property
    def state(self):
        """The tasks state."""
        return self._state
    status = state

    @property
    def traceback(self):
        """The traceback if the task failed."""
        return self._traceback

    @property
    def supports_native_join(self):
        return False