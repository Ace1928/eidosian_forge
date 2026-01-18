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
@unix_only_sock_test
@non_macos_sock_test
@pytest.mark.flaky(reruns=3, reruns_delay=2)
def test_peercreds_unix_sock(http_request_timeout, peercreds_enabled_server):
    """Check that ``PEERCRED`` lookup works when enabled."""
    httpserver = peercreds_enabled_server
    bind_addr = httpserver.bind_addr
    if isinstance(bind_addr, bytes):
        bind_addr = bind_addr.decode()
    quoted = urllib.parse.quote(bind_addr, safe='')
    unix_base_uri = 'http+unix://{quoted}'.format(**locals())
    expected_peercreds = (os.getpid(), os.getuid(), os.getgid())
    expected_peercreds = '|'.join(map(str, expected_peercreds))
    with requests_unixsocket.monkeypatch():
        peercreds_resp = requests.get(unix_base_uri + PEERCRED_IDS_URI, timeout=http_request_timeout)
        peercreds_resp.raise_for_status()
        assert peercreds_resp.text == expected_peercreds
        peercreds_text_resp = requests.get(unix_base_uri + PEERCRED_TEXTS_URI, timeout=http_request_timeout)
        assert peercreds_text_resp.status_code == 500