from __future__ import unicode_literals
from typing import cast
from .. import _url
from .common import HyperlinkTestCase
from .._url import register_scheme, URL, DecodedURL
def test_register_invalid_uses_netloc(self):
    with self.assertRaises(ValueError):
        register_scheme('lol', uses_netloc=cast(bool, object()))