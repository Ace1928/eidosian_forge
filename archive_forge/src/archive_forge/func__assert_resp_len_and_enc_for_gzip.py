import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
def _assert_resp_len_and_enc_for_gzip(self, uri):
    """
        Test that after querying gzipped content it's remains valid in
        cache and available non-gzipped as well.
        """
    ACCEPT_GZIP_HEADERS = [('Accept-Encoding', 'gzip')]
    content_len = None
    for _ in range(3):
        self.getPage(uri, method='GET', headers=ACCEPT_GZIP_HEADERS)
        if content_len is not None:
            self.assertHeader('Content-Length', content_len)
            self.assertHeader('Content-Encoding', 'gzip')
        content_len = dict(self.headers)['Content-Length']
    self.getPage(uri, method='GET')
    self.assertNoHeader('Content-Encoding')
    self.assertNoHeaderItemValue('Content-Length', content_len)