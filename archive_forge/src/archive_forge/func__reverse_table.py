import math
import struct
from contextlib import contextmanager
from numbers import Integral
from ..messages import BaseMessage, check_time
def _reverse_table(table):
    """Return value: key for dictionary."""
    return {value: key for key, value in table.items()}