from __future__ import unicode_literals
from typing import cast
from .. import _url
from .common import HyperlinkTestCase
from .._url import register_scheme, URL, DecodedURL
def test_register_no_netloc_scheme(self):
    register_scheme('noloctron', uses_netloc=False)
    u4 = URL(scheme='noloctron')
    u4 = u4.replace(path=('example', 'path'))
    assert u4.to_text() == 'noloctron:example/path'