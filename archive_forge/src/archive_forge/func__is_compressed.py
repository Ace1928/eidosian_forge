from __future__ import absolute_import
import asyncio
import functools
import aiohttp  # type: ignore
import six
import urllib3  # type: ignore
from google.auth import exceptions
from google.auth import transport
from google.auth.transport import requests
def _is_compressed(self):
    headers = self._response.headers
    return 'Content-Encoding' in headers and (headers['Content-Encoding'] == 'gzip' or headers['Content-Encoding'] == 'deflate')