from __future__ import unicode_literals
from typing import cast
from .. import _url
from .common import HyperlinkTestCase
from .._url import register_scheme, URL, DecodedURL
def test_register_no_quote_plus_scheme(self):
    register_scheme('keepplus', query_plus_is_space=False)
    plus_is_not_space = DecodedURL.from_text('keepplus://example.com/?q=a+b')
    plus_is_space = DecodedURL.from_text('https://example.com/?q=a+b')
    assert plus_is_not_space.get('q') == ['a+b']
    assert plus_is_space.get('q') == ['a b']