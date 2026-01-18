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
@pytest.fixture
def testing_server(raw_testing_server, monkeypatch):
    """Modify the "raw" base server to monitor the error_log messages.

    If you need to ignore a particular error message use the property
    ``testing_server.error_log.ignored_msgs`` by appending to the list
    the expected error messages.
    """
    monkeypatch.setattr(raw_testing_server, 'error_log', ErrorLogMonitor())
    yield raw_testing_server
    for c_msg, c_level, c_traceback in raw_testing_server.error_log.calls:
        if c_level <= logging.WARNING:
            continue
        assert c_msg in raw_testing_server.error_log.ignored_msgs, ("Found error in the error log: message = '{c_msg}', level = '{c_level}'\n{c_traceback}".format(**locals()),)