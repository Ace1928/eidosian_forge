import array
import os
import socket
from warnings import warn
class NoFDError(RuntimeError):
    """Raised by :class:`FileDescriptor` methods if it was already closed/converted
    """
    pass