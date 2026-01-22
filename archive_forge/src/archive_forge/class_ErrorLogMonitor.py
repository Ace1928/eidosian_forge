import errno
import socket
import time
import logging
import traceback as traceback_
from collections import namedtuple
import http.client
import urllib.request
import pytest
from jaraco.text import trim, unwrap
from cheroot.test import helper, webtest
from cheroot._compat import IS_CI, IS_MACOS, IS_PYPY, IS_WINDOWS
import cheroot.server
class ErrorLogMonitor:
    """Mock class to access the server error_log calls made by the server."""
    ErrorLogCall = namedtuple('ErrorLogCall', ['msg', 'level', 'traceback'])

    def __init__(self):
        """Initialize the server error log monitor/interceptor.

        If you need to ignore a particular error message use the property
        ``ignored_msgs`` by appending to the list the expected error messages.
        """
        self.calls = []
        self.ignored_msgs = []

    def __call__(self, msg='', level=logging.INFO, traceback=False):
        """Intercept the call to the server error_log method."""
        if traceback:
            tblines = traceback_.format_exc()
        else:
            tblines = ''
        self.calls.append(ErrorLogMonitor.ErrorLogCall(msg, level, tblines))