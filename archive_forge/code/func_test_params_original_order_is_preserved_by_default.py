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
def test_params_original_order_is_preserved_by_default(self):
    param_ordered_dict = collections.OrderedDict((('z', 1), ('a', 1), ('k', 1), ('d', 1)))
    session = requests.Session()
    request = requests.Request('GET', 'http://example.com/', params=param_ordered_dict)
    prep = session.prepare_request(request)
    assert prep.url == 'http://example.com/?z=1&a=1&k=1&d=1'