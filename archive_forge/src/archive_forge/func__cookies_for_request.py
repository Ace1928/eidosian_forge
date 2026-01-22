import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def _cookies_for_request(self, request):
    """Return a list of cookies to be returned to server."""
    cookies = []
    for domain in self._cookies.keys():
        cookies.extend(self._cookies_for_domain(domain, request))
    return cookies