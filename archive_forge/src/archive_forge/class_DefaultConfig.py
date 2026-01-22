from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
class DefaultConfig(Config):
    """A default configuration for tracing of messages in breezy.

    This implements the context manager protocol.
    """

    def __enter__(self):
        self._original_filename = _brz_log_filename
        self._original_state = enable_default_logging()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pop_log_file(self._original_state)
        global _brz_log_filename
        _brz_log_filename = self._original_filename
        return False