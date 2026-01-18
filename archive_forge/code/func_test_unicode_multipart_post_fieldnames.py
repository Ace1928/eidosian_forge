from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
def test_unicode_multipart_post_fieldnames(self, httpbin):
    filename = os.path.splitext(__file__)[0] + '.py'
    r = requests.Request(method='POST', url=httpbin('post'), data={'stuff'.encode('utf-8'): 'elixr'}, files={'file': ('test_requests.py', open(filename, 'rb'))})
    prep = r.prepare()
    assert b'name="stuff"' in prep.body
    assert b'name="b\'stuff\'"' not in prep.body