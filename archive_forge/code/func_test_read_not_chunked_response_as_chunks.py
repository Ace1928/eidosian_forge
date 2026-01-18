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
def test_read_not_chunked_response_as_chunks(self):
    fp = BytesIO(b'foo')
    resp = HTTPResponse(fp, preload_content=False)
    r = resp.read_chunked()
    with pytest.raises(ResponseNotChunked):
        next(r)