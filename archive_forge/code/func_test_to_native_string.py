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
@pytest.mark.parametrize('value, expected', (('T', 'T'), (b'T', 'T'), (u'T', 'T')))
def test_to_native_string(value, expected):
    assert to_native_string(value) == expected