import errno
import functools
import fcntl
import os
import struct
import threading
from . import exceptions
from . import _error_translation as errors
from .bindings import libzfs_core
from ._constants import MAXNAMELEN
from .ctypes import int32_t
from ._nvlist import nvlist_in, nvlist_out
def _uncommitted(depends_on=None):
    """
    Mark an API function as being an uncommitted extension that might not be
    available.

    :param function depends_on: the function that would be checked
                                instead of a decorated function.
                                For example, if the decorated function uses
                                another uncommitted function.

    This decorator transforms a decorated function to raise
    :exc:`NotImplementedError` if the C libzfs_core library does not provide
    a function with the same name as the decorated function.

    The optional `depends_on` parameter can be provided if the decorated
    function does not directly call the C function but instead calls another
    Python function that follows the typical convention.
    One example is :func:`lzc_list_snaps` that calls :func:`lzc_list` that
    calls ``lzc_list`` in libzfs_core.

    This decorator is implemented using :func:`is_supported`.
    """

    def _uncommitted_decorator(func, depends_on=depends_on):

        @functools.wraps(func)
        def _f(*args, **kwargs):
            if not is_supported(_f):
                raise NotImplementedError(func.__name__)
            return func(*args, **kwargs)
        if depends_on is not None:
            _f._check_func = depends_on
        return _f
    return _uncommitted_decorator