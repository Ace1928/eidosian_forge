import threading
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
class DefaultStack(threading.local):
    """A thread-local stack of objects for providing implicit defaults."""

    def __init__(self):
        super().__init__()
        self._enforce_nesting = True
        self.stack = []

    def get_default(self):
        return self.stack[-1] if self.stack else None

    def reset(self):
        self.stack = []

    def is_cleared(self):
        return not self.stack

    @property
    def enforce_nesting(self):
        return self._enforce_nesting

    @enforce_nesting.setter
    def enforce_nesting(self, value):
        self._enforce_nesting = value

    @tf_contextlib.contextmanager
    def get_controller(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            if self.stack:
                if self._enforce_nesting:
                    if self.stack[-1] is not default:
                        raise AssertionError('Nesting violated for default stack of %s objects' % type(default))
                    self.stack.pop()
                else:
                    self.stack.remove(default)