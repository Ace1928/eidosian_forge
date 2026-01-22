from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class RemoteConnectionError(OperationFailed):
    """Operations encountered remote connection trouble."""
    default_message = 'remote connection error'