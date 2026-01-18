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
@pytest.mark.skipif(not IS_UID_GID_RESOLVABLE, reason='Modules `grp` and `pwd` are not available under the current platform')
@unix_only_sock_test
@non_macos_sock_test
def test_peercreds_unix_sock_with_lookup(http_request_timeout, peercreds_enabled_server):
    """Check that ``PEERCRED`` resolution works when enabled."""
    httpserver = peercreds_enabled_server
    httpserver.peercreds_resolve_enabled = True
    bind_addr = httpserver.bind_addr
    if isinstance(bind_addr, bytes):
        bind_addr = bind_addr.decode()
    quoted = urllib.parse.quote(bind_addr, safe='')
    unix_base_uri = 'http+unix://{quoted}'.format(**locals())
    import grp
    import pwd
    expected_textcreds = (pwd.getpwuid(os.getuid()).pw_name, grp.getgrgid(os.getgid()).gr_name)
    expected_textcreds = '!'.join(map(str, expected_textcreds))
    with requests_unixsocket.monkeypatch():
        peercreds_text_resp = requests.get(unix_base_uri + PEERCRED_TEXTS_URI, timeout=http_request_timeout)
        peercreds_text_resp.raise_for_status()
        assert peercreds_text_resp.text == expected_textcreds