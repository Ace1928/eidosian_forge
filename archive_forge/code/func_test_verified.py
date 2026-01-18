import datetime
import json
import logging
import os.path
import shutil
import ssl
import sys
import tempfile
import warnings
from test import (
import mock
import pytest
import trustme
import urllib3.util as util
from dummyserver.server import (
from dummyserver.testcase import HTTPSDummyServerTestCase
from urllib3 import HTTPSConnectionPool
from urllib3.connection import RECENT_DATE, VerifiedHTTPSConnection
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.util.timeout import Timeout
from .. import has_alpn
def test_verified(self):
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_REQUIRED', ca_certs=DEFAULT_CA) as https_pool:
        conn = https_pool._new_conn()
        assert conn.__class__ == VerifiedHTTPSConnection
        with warnings.catch_warnings(record=True) as w:
            r = https_pool.request('GET', '/')
            assert r.status == 200
        if self.tls_protocol_deprecated():
            w = [x for x in w if x.category != DeprecationWarning]
        if sys.version_info >= (2, 7, 9) or util.IS_PYOPENSSL or util.IS_SECURETRANSPORT:
            assert w == []
        else:
            assert len(w) > 1
            assert any((x.category == InsecureRequestWarning for x in w))