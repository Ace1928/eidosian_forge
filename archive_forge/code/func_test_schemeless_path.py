from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_schemeless_path(self):
    """See issue #4"""
    u1 = URL.from_text('urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob')
    u2 = URL.from_text(u1.to_text())
    assert u1 == u2
    u3 = URL.from_text(u1.to_iri().to_text())
    assert u1 == u3
    assert u2 == u3
    u4 = URL.from_text('first-segment/urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob')
    u5 = u4.to_iri()
    assert u5.to_text() == 'first-segment/urn:ietf:wg:oauth:2.0:oob'
    u6 = URL.from_text(u5.to_text()).to_uri()
    assert u5 == u6