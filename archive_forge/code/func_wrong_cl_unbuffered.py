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
def wrong_cl_unbuffered(req, resp):
    """Render unbuffered response with invalid length value."""
    resp.headers['Content-Length'] = '5'
    return ['I too', ' have too many bytes']