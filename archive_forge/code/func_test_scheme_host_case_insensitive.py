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
def test_scheme_host_case_insensitive(self):
    """Assert that upper-case schemes and hosts are normalized."""
    with proxy_from_url(self.proxy_url.upper(), ca_certs=DEFAULT_CA) as http:
        r = http.request('GET', '%s/' % self.http_url.upper())
        assert r.status == 200
        r = http.request('GET', '%s/' % self.https_url.upper())
        assert r.status == 200