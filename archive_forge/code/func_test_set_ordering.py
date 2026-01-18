from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_set_ordering(self):
    url = URL.from_text('http://example.com/?a=b&c')
    url = url.set('x', 'x')
    url = url.add('x', 'y')
    assert url.to_text() == 'http://example.com/?a=b&x=x&c&x=y'