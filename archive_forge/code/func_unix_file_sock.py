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
@pytest.fixture
def unix_file_sock():
    """Yield a unix file socket."""
    tmp_sock_fh, tmp_sock_fname = tempfile.mkstemp()
    yield tmp_sock_fname
    os.close(tmp_sock_fh)
    os.unlink(tmp_sock_fname)