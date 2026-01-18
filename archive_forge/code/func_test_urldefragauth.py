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
@pytest.mark.parametrize('url, expected', (('http://u:p@example.com/path?a=1#test', 'http://example.com/path?a=1'), ('http://example.com/path', 'http://example.com/path'), ('//u:p@example.com/path', '//example.com/path'), ('//example.com/path', '//example.com/path'), ('example.com/path', '//example.com/path'), ('scheme:u:p@example.com/path', 'scheme://example.com/path')))
def test_urldefragauth(url, expected):
    assert urldefragauth(url) == expected