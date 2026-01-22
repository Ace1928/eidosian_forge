import collections
import inspect
import threading
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack
class StackTraceTransform(object):
    """Base class for stack trace transformation functions."""
    _stack_dict = None
    _thread_key = None

    def __enter__(self):
        if self._thread_key is None:
            self._thread_key = _get_thread_key()
        else:
            assert self._thread_key == _get_thread_key(), 'Shared across threads?'
        stack = self._stack_dict[self._thread_key]
        self.parent = stack[-1]
        stack.append(self)
        self.update()
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        top = self._stack_dict[self._thread_key].pop()
        assert top is self, 'Concurrent access?'

    def update(self):
        raise NotImplementedError('subclasses need to override this')