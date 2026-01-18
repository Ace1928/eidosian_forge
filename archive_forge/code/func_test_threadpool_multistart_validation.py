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
def test_threadpool_multistart_validation(monkeypatch):
    """Test for ThreadPool multi-start behavior.

    Tests that when calling start() on a ThreadPool multiple times raises a
    :exc:`RuntimeError`
    """
    monkeypatch.setattr(ThreadPool, '_spawn_worker', lambda _: types.SimpleNamespace(ready=True))
    tp = ThreadPool(server=None)
    tp.start()
    with pytest.raises(RuntimeError, match='Threadpools can only be started once.'):
        tp.start()