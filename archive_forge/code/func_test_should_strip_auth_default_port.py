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
@pytest.mark.parametrize('old_uri, new_uri', (('https://example.com:443/foo', 'https://example.com/bar'), ('http://example.com:80/foo', 'http://example.com/bar'), ('https://example.com/foo', 'https://example.com:443/bar'), ('http://example.com/foo', 'http://example.com:80/bar')))
def test_should_strip_auth_default_port(self, old_uri, new_uri):
    s = requests.Session()
    assert not s.should_strip_auth(old_uri, new_uri)