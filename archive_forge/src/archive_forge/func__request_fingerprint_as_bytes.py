import hashlib
import json
import warnings
from typing import (
from urllib.parse import urlunparse
from weakref import WeakKeyDictionary
from w3lib.http import basic_auth_header
from w3lib.url import canonicalize_url
from scrapy import Request, Spider
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_bytes, to_unicode
def _request_fingerprint_as_bytes(*args: Any, **kwargs: Any) -> bytes:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return bytes.fromhex(request_fingerprint(*args, **kwargs))