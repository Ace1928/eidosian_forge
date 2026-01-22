import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def _oauth_escape(val: Union[str, bytes]) -> str:
    if isinstance(val, unicode_type):
        val = val.encode('utf-8')
    return urllib.parse.quote(val, safe='~')