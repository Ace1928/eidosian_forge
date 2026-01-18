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
@pytest.mark.parametrize('var,url,proxy', [('http_proxy', 'http://example.com', 'socks5://proxy.com:9876'), ('https_proxy', 'https://example.com', 'socks5://proxy.com:9876'), ('all_proxy', 'http://example.com', 'socks5://proxy.com:9876'), ('all_proxy', 'https://example.com', 'socks5://proxy.com:9876')])
def test_proxy_env_vars_override_default(var, url, proxy):
    session = requests.Session()
    prep = PreparedRequest()
    prep.prepare(method='GET', url=url)
    kwargs = {var: proxy}
    scheme = urlparse(url).scheme
    with override_environ(**kwargs):
        proxies = session.rebuild_proxies(prep, {})
        assert scheme in proxies
        assert proxies[scheme] == proxy