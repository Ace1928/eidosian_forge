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
def _close_kernel_socket(self):
    monkeypatch.setattr(self, 'socket', mocker.mock_module.Mock(wraps=self.socket))
    if exc_instance is not None:
        monkeypatch.setattr(self.socket, 'shutdown', mocker.mock_module.Mock(side_effect=exc_instance))
    _close_kernel_socket.fin_spy = mocker.spy(self.socket, 'shutdown')
    try:
        old_close_kernel_socket(self)
    except simulated_exception:
        _close_kernel_socket.exception_leaked = True
    else:
        _close_kernel_socket.exception_leaked = False