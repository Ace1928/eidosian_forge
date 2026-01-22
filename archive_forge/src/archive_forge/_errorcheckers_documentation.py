import errno
import os
from ctypes import get_errno
Error checker to check the system ``errno`` as returned by
    :func:`ctypes.get_errno()`.

    If ``result`` is a null pointer, an exception according to this errno is
    raised.  Otherwise nothing happens.

    