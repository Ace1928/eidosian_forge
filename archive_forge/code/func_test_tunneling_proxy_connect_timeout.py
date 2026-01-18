import json
import os.path
import shutil
import socket
import ssl
import sys
import tempfile
import warnings
from test import (
import pytest
import trustme
from dummyserver.server import DEFAULT_CA, HAS_IPV6, get_unreachable_address
from dummyserver.testcase import HTTPDummyProxyTestCase, IPv6HTTPDummyProxyTestCase
from urllib3._collections import HTTPHeaderDict
from urllib3.connectionpool import VerifiedHTTPSConnection, connection_from_url
from urllib3.exceptions import (
from urllib3.poolmanager import ProxyManager, proxy_from_url
from urllib3.util import Timeout
from urllib3.util.ssl_ import create_urllib3_context
from .. import TARPIT_HOST, requires_network
@requires_network
@pytest.mark.parametrize(['proxy_scheme', 'target_scheme'], [('http', 'https'), ('https', 'https')])
def test_tunneling_proxy_connect_timeout(self, proxy_scheme, target_scheme):
    _should_skip_https_in_https(proxy_scheme, target_scheme)
    proxy_url = '%s://%s' % (proxy_scheme, TARPIT_HOST)
    target_url = self.https_url if target_scheme == 'https' else self.http_url
    with proxy_from_url(proxy_url, ca_certs=DEFAULT_CA, timeout=SHORT_TIMEOUT) as proxy:
        with pytest.raises(MaxRetryError) as e:
            proxy.request('GET', target_url)
        assert type(e.value.reason) == ConnectTimeoutError