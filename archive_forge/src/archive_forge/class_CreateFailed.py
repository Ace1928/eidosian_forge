from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class CreateFailed(FSError):
    """Filesystem could not be created."""
    default_message = 'unable to create filesystem, {details}'

    def __init__(self, msg=None, exc=None):
        self._msg = msg or self.default_message
        self.details = '' if exc is None else text_type(exc)
        self.exc = exc

    @classmethod
    def catch_all(cls, func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except cls:
                raise
            except Exception as e:
                raise cls(exc=e)
        return new_func

    def __reduce__(self):
        return (type(self), (self._msg, self.exc))