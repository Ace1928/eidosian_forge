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
def test_precedence(self):
    content = '\n        <?xml version="1.0" encoding="XML"?>\n        <meta charset="HTML5">\n        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />\n        '.strip()
    assert get_encodings_from_content(content) == ['HTML5', 'HTML4', 'XML']