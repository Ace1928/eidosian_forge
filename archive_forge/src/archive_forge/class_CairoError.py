import sys
from ctypes.util import find_library
from . import constants
from .ffi import ffi
from .surfaces import (  # noqa isort:skip
from .patterns import (  # noqa isort:skip
from .fonts import (  # noqa isort:skip
from .context import Context  # noqa isort:skip
from .matrix import Matrix  # noqa isort:skip
from .constants import *  # noqa isort:skip
class CairoError(Exception):
    """Raised when cairo returns an error status."""

    def __init__(self, message, status):
        super(CairoError, self).__init__(message)
        self.status = status