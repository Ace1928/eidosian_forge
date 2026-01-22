from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _handlePause(_pause):
    """
    A helper function for performing a pause at the end of a PyAutoGUI function based on some settings.

    If ``_pause`` is ``True``, then sleep for ``PAUSE`` seconds (the global pause setting).
    """
    if _pause:
        assert isinstance(PAUSE, int) or isinstance(PAUSE, float)
        time.sleep(PAUSE)