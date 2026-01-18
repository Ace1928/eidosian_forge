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
def test_proxy_verified(self):
    with proxy_from_url(self.proxy_url, cert_reqs='REQUIRED', ca_certs=self.bad_ca_path) as http:
        https_pool = http._new_pool('https', self.https_host, self.https_port)
        with pytest.raises(MaxRetryError) as e:
            https_pool.request('GET', '/', retries=0)
        assert isinstance(e.value.reason, SSLError)
        assert 'certificate verify failed' in str(e.value.reason), "Expected 'certificate verify failed', instead got: %r" % e.value.reason
        http = proxy_from_url(self.proxy_url, cert_reqs='REQUIRED', ca_certs=DEFAULT_CA)
        https_pool = http._new_pool('https', self.https_host, self.https_port)
        conn = https_pool._new_conn()
        assert conn.__class__ == VerifiedHTTPSConnection
        https_pool.request('GET', '/')
        http = proxy_from_url(self.proxy_url, cert_reqs='REQUIRED', ca_certs=DEFAULT_CA)
        https_fail_pool = http._new_pool('https', '127.0.0.1', self.https_port)
        with pytest.raises(MaxRetryError) as e:
            https_fail_pool.request('GET', '/', retries=0)
        assert isinstance(e.value.reason, SSLError)
        assert "doesn't match" in str(e.value.reason)