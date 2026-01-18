import contextlib
import re
import socket
import ssl
import zlib
from base64 import b64decode
from io import BufferedReader, BytesIO, TextIOWrapper
from test import onlyBrotlipy
import mock
import pytest
import six
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.response import HTTPResponse, brotli
from urllib3.util.response import is_fp_closed
from urllib3.util.retry import RequestHistory, Retry
def test_reference_read(self):
    fp = BytesIO(b'foo')
    r = HTTPResponse(fp, preload_content=False)
    assert r.read(1) == b'f'
    assert r.read(2) == b'oo'
    assert r.read() == b''
    assert r.read() == b''