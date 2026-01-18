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
def test_super_len_correctly_calculates_len_of_partially_read_file(self):
    """Ensure that we handle partially consumed file like objects."""
    s = StringIO.StringIO()
    s.write('foobarbogus')
    assert super_len(s) == 0