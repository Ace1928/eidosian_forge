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
def test_response_chunk_size_type(self):
    """Ensure that chunk_size is passed as None or an integer, otherwise
        raise a TypeError.
        """
    r = requests.Response()
    r.raw = io.BytesIO(b'the content')
    chunks = r.iter_content(1)
    assert all((len(chunk) == 1 for chunk in chunks))
    r = requests.Response()
    r.raw = io.BytesIO(b'the content')
    chunks = r.iter_content(None)
    assert list(chunks) == [b'the content']
    r = requests.Response()
    r.raw = io.BytesIO(b'the content')
    with pytest.raises(TypeError):
        chunks = r.iter_content('1024')