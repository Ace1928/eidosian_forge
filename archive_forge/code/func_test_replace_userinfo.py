from __future__ import unicode_literals
from typing import Dict, Union
from .. import DecodedURL, URL
from .._url import _percent_decode
from .common import HyperlinkTestCase
def test_replace_userinfo(self):
    durl = DecodedURL.from_text(TOTAL_URL)
    with self.assertRaises(ValueError):
        durl.replace(userinfo=('user', 'pw', 'thiswillcauseafailure'))
    return