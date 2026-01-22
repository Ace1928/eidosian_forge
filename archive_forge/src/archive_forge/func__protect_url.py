import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _protect_url(self, url):
    """
        Function that passes a URL through `_html_escape_url` to remove any nasty characters,
        and then hashes the now "safe" URL to prevent other safety mechanisms from tampering
        with it (eg: escaping "&" in URL parameters)
        """
    data_url = self._data_url_re.match(url)
    charset = None
    if data_url is not None:
        mime = data_url.group('mime') or ''
        if mime.startswith('image/') and data_url.group('token') == ';base64':
            charset = 'base64'
    url = _html_escape_url(url, safe_mode=self.safe_mode, charset=charset)
    key = _hash_text(url)
    self._escape_table[url] = key
    return key