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
@pytest.mark.parametrize('uri, expected', (('http://example.com/fiz?buz=%25ppicture', 'http://example.com/fiz?buz=%25ppicture'), ('http://example.com/fiz?buz=%ppicture', 'http://example.com/fiz?buz=%25ppicture')))
def test_requote_uri_with_unquoted_percents(uri, expected):
    """See: https://github.com/psf/requests/issues/2356"""
    assert requote_uri(uri) == expected