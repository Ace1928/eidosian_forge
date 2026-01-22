import functools
import logging
import os
import pipes
import shutil
import sys
import tempfile
import time
import unittest
from humanfriendly.compat import StringIO
from humanfriendly.text import random_string
class CaptureBuffer(StringIO):
    """
    Helper for :class:`CaptureOutput` to provide an easy to use API.

    The two methods defined by this subclass were specifically chosen to match
    the names of the methods provided by my :pypi:`capturer` package which
    serves a similar role as :class:`CaptureOutput` but knows how to simulate
    an interactive terminal (tty).
    """

    def get_lines(self):
        """Get the contents of the buffer split into separate lines."""
        return self.get_text().splitlines()

    def get_text(self):
        """Get the contents of the buffer as a Unicode string."""
        return self.getvalue()