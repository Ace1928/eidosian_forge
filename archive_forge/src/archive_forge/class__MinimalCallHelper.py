from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
class _MinimalCallHelper(object):
    """
    A call helper object for the "minimal" calling convention.
    User exceptions are represented as integer codes and stored in
    a mapping for retrieval from the caller.
    """

    def __init__(self):
        self.exceptions = {}

    def _add_exception(self, exc, exc_args, locinfo):
        """
        Add a new user exception to this helper. Returns an integer that can be
        used to refer to the added exception in future.

        Parameters
        ----------
        exc :
            exception type
        exc_args : None or tuple
            exception args
        locinfo : tuple
            location information
        """
        exc_id = len(self.exceptions) + FIRST_USEREXC
        self.exceptions[exc_id] = (exc, exc_args, locinfo)
        return exc_id

    def get_exception(self, exc_id):
        """
        Get information about a user exception. Returns a tuple of
        (exception type, exception args, location information).

        Parameters
        ----------
        id : integer
            The ID of the exception to look up
        """
        try:
            return self.exceptions[exc_id]
        except KeyError:
            msg = 'unknown error %d in native function' % exc_id
            exc = SystemError
            exc_args = (msg,)
            locinfo = None
            return (exc, exc_args, locinfo)