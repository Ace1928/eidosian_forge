import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
def _show_any_warnings(self):
    """If there were any messages since the last reset, show them
        as a warning. Otherwise do nothing. Also resets the messages.
        """
    if self._messages:
        logger.warning('imageio.freeimage warning: ' + self._get_error_message())
        self._reset_log()