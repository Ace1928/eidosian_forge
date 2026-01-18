import os
import queue
import socket
import tempfile
import threading
import types
import uuid
import urllib.parse  # noqa: WPS301
import pytest
import requests
import requests_unixsocket
from pypytools.gc.custom import DefaultGc
from .._compat import bton, ntob
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS, SYS_PLATFORM
from ..server import IS_UID_GID_RESOLVABLE, Gateway, HTTPServer
from ..workers.threadpool import ThreadPool
from ..testing import (
def test_serving_is_false_and_stop_returns_after_ctrlc():
    """Check that stop() interrupts running of serve()."""
    httpserver = HTTPServer(bind_addr=(ANY_INTERFACE_IPV4, EPHEMERAL_PORT), gateway=Gateway)
    httpserver.prepare()

    def raise_keyboard_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    httpserver._connections._selector.select = raise_keyboard_interrupt
    serve_thread = threading.Thread(target=httpserver.serve)
    serve_thread.start()
    serve_thread.join(httpserver.expiration_interval * (4 if IS_SLOW_ENV else 2))
    assert not serve_thread.is_alive()
    assert not httpserver._connections._serving
    httpserver.stop()