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
class MockChunkedEncodingWithoutCRLFOnEnd(MockChunkedEncodingResponse):

    def _encode_chunk(self, chunk):
        return '%X\r\n%s%s' % (len(chunk), chunk.decode(), '\r\n' if len(chunk) > 0 else '')