import functools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import http.client
import OpenSSL.SSL
import pytest
import requests
import trustme
from .._compat import bton, ntob, ntou
from .._compat import IS_ABOVE_OPENSSL10, IS_CI, IS_PYPY
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS
from ..server import HTTPServer, get_ssl_adapter_class
from ..testing import (
from ..wsgi import Gateway_10
@pytest.fixture
def thread_exceptions():
    """Provide a list of uncaught exceptions from threads via a fixture.

    Only catches exceptions on Python 3.8+.
    The list contains: ``(type, str(value), str(traceback))``
    """
    exceptions = []
    orig_hook = getattr(threading, 'excepthook', None)
    if orig_hook is not None:
        threading.excepthook = functools.partial(_thread_except_hook, exceptions)
    try:
        yield exceptions
    finally:
        if orig_hook is not None:
            threading.excepthook = orig_hook