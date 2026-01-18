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
@pytest.mark.parametrize('input, expected', ((b'http+unix://%2Fvar%2Frun%2Fsocket/path%7E', u'http+unix://%2Fvar%2Frun%2Fsocket/path~'), (u'http+unix://%2Fvar%2Frun%2Fsocket/path%7E', u'http+unix://%2Fvar%2Frun%2Fsocket/path~'), (b'mailto:user@example.org', u'mailto:user@example.org'), (u'mailto:user@example.org', u'mailto:user@example.org'), (b'data:SSDimaUgUHl0aG9uIQ==', u'data:SSDimaUgUHl0aG9uIQ==')))
def test_url_mutation(self, input, expected):
    """
        This test validates that we correctly exclude some URLs from
        preparation, and that we handle others. Specifically, it tests that
        any URL whose scheme doesn't begin with "http" is left alone, and
        those whose scheme *does* begin with "http" are mutated.
        """
    r = requests.Request('GET', url=input)
    p = r.prepare()
    assert p.url == expected