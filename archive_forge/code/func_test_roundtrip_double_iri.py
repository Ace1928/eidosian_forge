from __future__ import unicode_literals
import sys
import socket
from typing import Any, Iterable, Optional, Text, Tuple, cast
from .common import HyperlinkTestCase
from .. import URL, URLParseError
from .._url import inet_pton, SCHEME_PORT_MAP
def test_roundtrip_double_iri(self):
    for test in ROUNDTRIP_TESTS:
        url = URL.from_text(test)
        iri = url.to_iri()
        double_iri = iri.to_iri()
        assert iri == double_iri
        iri_text = iri.to_text(with_password=True)
        double_iri_text = double_iri.to_text(with_password=True)
        assert iri_text == double_iri_text
    return