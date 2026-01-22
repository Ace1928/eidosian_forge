import os
import socket
from errno import errorcode
from functools import partial, wraps
from itertools import chain, count
from sys import platform
from weakref import WeakValueDictionary
from OpenSSL._util import (
from OpenSSL.crypto import (
class _CallbackExceptionHelper:
    """
    A base class for wrapper classes that allow for intelligent exception
    handling in OpenSSL callbacks.

    :ivar list _problems: Any exceptions that occurred while executing in a
        context where they could not be raised in the normal way.  Typically
        this is because OpenSSL has called into some Python code and requires a
        return value.  The exceptions are saved to be raised later when it is
        possible to do so.
    """

    def __init__(self):
        self._problems = []

    def raise_if_problem(self):
        """
        Raise an exception from the OpenSSL error queue or that was previously
        captured whe running a callback.
        """
        if self._problems:
            try:
                _raise_current_error()
            except Error:
                pass
            raise self._problems.pop(0)