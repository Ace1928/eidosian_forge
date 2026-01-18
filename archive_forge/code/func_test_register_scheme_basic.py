from __future__ import unicode_literals
from typing import cast
from .. import _url
from .common import HyperlinkTestCase
from .._url import register_scheme, URL, DecodedURL
def test_register_scheme_basic(self):
    register_scheme('deltron', uses_netloc=True, default_port=3030)
    u1 = URL.from_text('deltron://example.com')
    assert u1.scheme == 'deltron'
    assert u1.port == 3030
    assert u1.uses_netloc is True
    u2 = URL.from_text('deltron:')
    u2 = u2.replace(host='example.com')
    assert u2.to_text() == 'deltron://example.com'
    u3 = URL.from_text('deltron://example.com:3030')
    assert u3.to_text() == 'deltron://example.com'
    register_scheme('nonetron', default_port=3031)
    u4 = URL(scheme='nonetron')
    u4 = u4.replace(host='example.com')
    assert u4.to_text() == 'nonetron://example.com'