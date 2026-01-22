from __future__ import annotations
import ctypes
from pandas._config.config import OptionError
from pandas._libs.tslibs import (
from pandas.util.version import InvalidVersion
class NoBufferPresent(Exception):
    """
    Exception is raised in _get_data_buffer to signal that there is no requested buffer.
    """