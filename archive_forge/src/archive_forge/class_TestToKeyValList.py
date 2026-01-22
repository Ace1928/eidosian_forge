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
class TestToKeyValList:

    @pytest.mark.parametrize('value, expected', (([('key', 'val')], [('key', 'val')]), ((('key', 'val'),), [('key', 'val')]), ({'key': 'val'}, [('key', 'val')]), (None, None)))
    def test_valid(self, value, expected):
        assert to_key_val_list(value) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            to_key_val_list('string')