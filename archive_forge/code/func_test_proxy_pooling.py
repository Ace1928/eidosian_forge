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
def test_proxy_pooling(self):
    with proxy_from_url(self.proxy_url, cert_reqs='NONE') as http:
        for x in range(2):
            http.urlopen('GET', self.http_url)
        assert len(http.pools) == 1
        for x in range(2):
            http.urlopen('GET', self.http_url_alt)
        assert len(http.pools) == 1
        for x in range(2):
            http.urlopen('GET', self.https_url)
        assert len(http.pools) == 2
        for x in range(2):
            http.urlopen('GET', self.https_url_alt)
        assert len(http.pools) == 3