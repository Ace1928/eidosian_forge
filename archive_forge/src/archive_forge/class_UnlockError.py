from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
class UnlockError(Error):
    """
    Base class for errors arising from attempts to release the lock.

    >>> try:
    ...   raise UnlockError
    ... except Error:
    ...   pass
    """
    pass