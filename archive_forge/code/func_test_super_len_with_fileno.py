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
def test_super_len_with_fileno(self):
    with open(__file__, 'rb') as f:
        length = super_len(f)
        file_data = f.read()
    assert length == len(file_data)