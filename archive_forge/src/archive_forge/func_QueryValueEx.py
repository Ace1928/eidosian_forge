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
def QueryValueEx(key, value_name):
    if key is ie_settings:
        if value_name == 'ProxyEnable':
            proxyEnableValues.rotate()
            return [proxyEnableValues[0]]
        elif value_name == 'ProxyOverride':
            return [override]