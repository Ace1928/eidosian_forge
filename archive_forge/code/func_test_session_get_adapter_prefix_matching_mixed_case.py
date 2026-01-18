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
def test_session_get_adapter_prefix_matching_mixed_case(self):
    mixed_case_prefix = 'hTtPs://eXamPle.CoM/MixEd_CAse_PREfix'
    url_matching_prefix = mixed_case_prefix + '/full_url'
    s = requests.Session()
    my_adapter = HTTPAdapter()
    s.mount(mixed_case_prefix, my_adapter)
    assert s.get_adapter(url_matching_prefix) is my_adapter