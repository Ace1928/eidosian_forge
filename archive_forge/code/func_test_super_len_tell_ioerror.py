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
@pytest.mark.parametrize('error', [IOError, OSError])
def test_super_len_tell_ioerror(self, error):
    """Ensure that if tell gives an IOError super_len doesn't fail"""

    class NoLenBoomFile(object):

        def tell(self):
            raise error()

        def seek(self, offset, whence):
            pass
    assert super_len(NoLenBoomFile()) == 0