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
class TestGuessFilename:

    @pytest.mark.parametrize('value', (1, type('Fake', (object,), {'name': 1})()))
    def test_guess_filename_invalid(self, value):
        assert guess_filename(value) is None

    @pytest.mark.parametrize('value, expected_type', ((b'value', compat.bytes), (b'value'.decode('utf-8'), compat.str)))
    def test_guess_filename_valid(self, value, expected_type):
        obj = type('Fake', (object,), {'name': value})()
        result = guess_filename(obj)
        assert result == value
        assert isinstance(result, expected_type)