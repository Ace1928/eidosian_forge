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
@pytest.mark.skipif(os.name != 'nt', reason='Test only on Windows')
@pytest.mark.parametrize('url, expected, override', (('http://192.168.0.1:5000/', True, None), ('http://192.168.0.1/', True, None), ('http://172.16.1.1/', True, None), ('http://172.16.1.1:5000/', True, None), ('http://localhost.localdomain:5000/v1.0/', True, None), ('http://172.16.1.22/', False, None), ('http://172.16.1.22:5000/', False, None), ('http://google.com:5000/v1.0/', False, None), ('http://mylocalhostname:5000/v1.0/', True, '<local>'), ('http://192.168.0.1/', False, '')))
def test_should_bypass_proxies_win_registry(url, expected, override, monkeypatch):
    """Tests for function should_bypass_proxies to check if proxy
    can be bypassed or not with Windows registry settings
    """
    if override is None:
        override = '192.168.*;127.0.0.1;localhost.localdomain;172.16.1.1'
    if compat.is_py3:
        import winreg
    else:
        import _winreg as winreg

    class RegHandle:

        def Close(self):
            pass
    ie_settings = RegHandle()
    proxyEnableValues = deque([1, '1'])

    def OpenKey(key, subkey):
        return ie_settings

    def QueryValueEx(key, value_name):
        if key is ie_settings:
            if value_name == 'ProxyEnable':
                proxyEnableValues.rotate()
                return [proxyEnableValues[0]]
            elif value_name == 'ProxyOverride':
                return [override]
    monkeypatch.setenv('http_proxy', '')
    monkeypatch.setenv('https_proxy', '')
    monkeypatch.setenv('ftp_proxy', '')
    monkeypatch.setenv('no_proxy', '')
    monkeypatch.setenv('NO_PROXY', '')
    monkeypatch.setattr(winreg, 'OpenKey', OpenKey)
    monkeypatch.setattr(winreg, 'QueryValueEx', QueryValueEx)
    assert should_bypass_proxies(url, None) == expected