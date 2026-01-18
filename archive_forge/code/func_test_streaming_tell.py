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
def test_streaming_tell(self):
    fp = BytesIO(b'foo')
    resp = HTTPResponse(fp, preload_content=False)
    stream = resp.stream(2, decode_content=False)
    position = 0
    position += len(next(stream))
    assert 2 == position
    assert position == resp.tell()
    position += len(next(stream))
    assert 3 == position
    assert position == resp.tell()
    with pytest.raises(StopIteration):
        next(stream)