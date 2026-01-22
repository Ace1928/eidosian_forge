import io
import os
import sys
import socket
import struct
import time
import tempfile
import itertools
from . import util
from . import AuthenticationError, BufferTooShort
from .context import reduction
def _validate_family(family):
    """
    Checks if the family is valid for the current environment.
    """
    if sys.platform != 'win32' and family == 'AF_PIPE':
        raise ValueError('Family %s is not recognized.' % family)
    if sys.platform == 'win32' and family == 'AF_UNIX':
        if not hasattr(socket, family):
            raise ValueError('Family %s is not recognized.' % family)