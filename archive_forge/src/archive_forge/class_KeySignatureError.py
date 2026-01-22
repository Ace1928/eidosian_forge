import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
class KeySignatureError(Exception):
    """ Raised when key cannot be converted from key/mode to key letter """
    pass