import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def add_cookie_header(self, request):
    """Add correct Cookie: header to request (urllib.request.Request object).

        The Cookie2 header is also added unless policy.hide_cookie2 is true.

        """
    _debug('add_cookie_header')
    self._cookies_lock.acquire()
    try:
        self._policy._now = self._now = int(time.time())
        cookies = self._cookies_for_request(request)
        attrs = self._cookie_attrs(cookies)
        if attrs:
            if not request.has_header('Cookie'):
                request.add_unredirected_header('Cookie', '; '.join(attrs))
        if self._policy.rfc2965 and (not self._policy.hide_cookie2) and (not request.has_header('Cookie2')):
            for cookie in cookies:
                if cookie.version != 1:
                    request.add_unredirected_header('Cookie2', '$Version="1"')
                    break
    finally:
        self._cookies_lock.release()
    self.clear_expired_cookies()