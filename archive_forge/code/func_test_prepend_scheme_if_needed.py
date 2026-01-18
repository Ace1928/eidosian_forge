import os
import copy
import filecmp
from io import BytesIO
import tarfile
import zipfile
from collections import deque
import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
from requests._internal_utils import unicode_is_ascii
from .compat import StringIO, cStringIO
@pytest.mark.parametrize('value, expected', (('example.com/path', 'http://example.com/path'), ('//example.com/path', 'http://example.com/path'), ('example.com:80', 'http://example.com:80'), ('http://user:pass@example.com/path?query', 'http://user:pass@example.com/path?query'), ('http://user@example.com/path?query', 'http://user@example.com/path?query')))
def test_prepend_scheme_if_needed(value, expected):
    assert prepend_scheme_if_needed(value, 'http') == expected