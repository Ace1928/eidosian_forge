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
@pytest.mark.parametrize('env_name, value', (('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain'), ('no_proxy', None), ('a_new_key', '192.168.0.0/24,127.0.0.1,localhost.localdomain'), ('a_new_key', None)))
def test_set_environ(env_name, value):
    """Tests set_environ will set environ values and will restore the environ."""
    environ_copy = copy.deepcopy(os.environ)
    with set_environ(env_name, value):
        assert os.environ.get(env_name) == value
    assert os.environ == environ_copy