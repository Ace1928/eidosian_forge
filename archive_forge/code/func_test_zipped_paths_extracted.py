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
def test_zipped_paths_extracted(self, tmpdir):
    zipped_py = tmpdir.join('test.zip')
    with zipfile.ZipFile(zipped_py.strpath, 'w') as f:
        f.write(__file__)
    _, name = os.path.splitdrive(__file__)
    zipped_path = os.path.join(zipped_py.strpath, name.lstrip('\\/'))
    extracted_path = extract_zipped_paths(zipped_path)
    assert extracted_path != zipped_path
    assert os.path.exists(extracted_path)
    assert filecmp.cmp(extracted_path, __file__)