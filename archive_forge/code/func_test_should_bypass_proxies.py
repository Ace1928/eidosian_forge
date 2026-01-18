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
@pytest.mark.parametrize('url, expected', (('http://192.168.0.1:5000/', True), ('http://192.168.0.1/', True), ('http://172.16.1.1/', True), ('http://172.16.1.1:5000/', True), ('http://localhost.localdomain:5000/v1.0/', True), ('http://google.com:6000/', True), ('http://172.16.1.12/', False), ('http://172.16.1.12:5000/', False), ('http://google.com:5000/v1.0/', False), ('file:///some/path/on/disk', True)))
def test_should_bypass_proxies(url, expected, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not
    """
    monkeypatch.setenv('no_proxy', '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1, google.com:6000')
    monkeypatch.setenv('NO_PROXY', '192.168.0.0/24,127.0.0.1,localhost.localdomain,172.16.1.1, google.com:6000')
    assert should_bypass_proxies(url, no_proxy=None) == expected