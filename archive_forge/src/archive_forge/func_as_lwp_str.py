import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def as_lwp_str(self, ignore_discard=True, ignore_expires=True):
    """Return cookies as a string of "\\n"-separated "Set-Cookie3" headers.

        ignore_discard and ignore_expires: see docstring for FileCookieJar.save

        """
    now = time.time()
    r = []
    for cookie in self:
        if not ignore_discard and cookie.discard:
            continue
        if not ignore_expires and cookie.is_expired(now):
            continue
        r.append('Set-Cookie3: %s' % lwp_cookie_str(cookie))
    return '\n'.join(r + [''])