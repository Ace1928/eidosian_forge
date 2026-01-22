from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
@six.python_2_unicode_compatible
class FSError(Exception):
    """Base exception for the `fs` module."""
    default_message = 'Unspecified error'

    def __init__(self, msg=None):
        self._msg = msg or self.default_message
        super(FSError, self).__init__()

    def __str__(self):
        """Return the error message."""
        msg = self._msg.format(**self.__dict__)
        return msg

    def __repr__(self):
        msg = self._msg.format(**self.__dict__)
        return '{}({!r})'.format(self.__class__.__name__, msg)