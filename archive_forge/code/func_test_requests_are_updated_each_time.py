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
def test_requests_are_updated_each_time(httpbin):
    session = RedirectSession([303, 307])
    prep = requests.Request('POST', httpbin('post')).prepare()
    r0 = session.send(prep)
    assert r0.request.method == 'POST'
    assert session.calls[-1] == SendCall((r0.request,), {})
    redirect_generator = session.resolve_redirects(r0, prep)
    default_keyword_args = {'stream': False, 'verify': True, 'cert': None, 'timeout': None, 'allow_redirects': False, 'proxies': {}}
    for response in redirect_generator:
        assert response.request.method == 'GET'
        send_call = SendCall((response.request,), default_keyword_args)
        assert session.calls[-1] == send_call