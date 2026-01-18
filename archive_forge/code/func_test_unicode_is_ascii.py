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
@pytest.mark.parametrize('value, expected', ((u'test', True), (u'æíöû', False), (u'ジェーピーニック', False)))
def test_unicode_is_ascii(value, expected):
    assert unicode_is_ascii(value) is expected