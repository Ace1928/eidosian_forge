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
def test_super_len_with_tell(self):
    foo = StringIO.StringIO('12345')
    assert super_len(foo) == 5
    foo.read(2)
    assert super_len(foo) == 3