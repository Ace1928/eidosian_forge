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
class TestIsValidCIDR:

    def test_valid(self):
        assert is_valid_cidr('192.168.1.0/24')

    @pytest.mark.parametrize('value', ('8.8.8.8', '192.168.1.0/a', '192.168.1.0/128', '192.168.1.0/-1', '192.168.1.999/24'))
    def test_invalid(self, value):
        assert not is_valid_cidr(value)