from __future__ import annotations
import string
import sys
from typing import Dict
from urllib.error import HTTPError
from urllib.parse import quote as urlquote
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, urlopen
from urllib.response import addinfourl

    This is a shim for `urlopen` that handles HTTP redirects with status code
    308 (Permanent Redirect).

    This function should be removed once all supported versions of Python
    handles the 308 HTTP status code.

    :param request: The request to open.
    :return: The response to the request.
    