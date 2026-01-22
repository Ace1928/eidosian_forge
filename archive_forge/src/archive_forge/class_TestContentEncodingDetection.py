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
class TestContentEncodingDetection:

    def test_none(self):
        encodings = get_encodings_from_content('')
        assert not len(encodings)

    @pytest.mark.parametrize('content', ('<meta charset="UTF-8">', '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">', '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />', '<?xml version="1.0" encoding="UTF-8"?>'))
    def test_pragmas(self, content):
        encodings = get_encodings_from_content(content)
        assert len(encodings) == 1
        assert encodings[0] == 'UTF-8'

    def test_precedence(self):
        content = '\n        <?xml version="1.0" encoding="XML"?>\n        <meta charset="HTML5">\n        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />\n        '.strip()
        assert get_encodings_from_content(content) == ['HTML5', 'HTML4', 'XML']