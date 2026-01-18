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
def test_ssl_unverified_with_ca_certs(self):
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_NONE', ca_certs=self.bad_ca_path) as pool:
        with mock.patch('warnings.warn') as warn:
            r = pool.request('GET', '/')
            assert r.status == 200
            assert warn.called
            calls = warn.call_args_list
            if self.tls_protocol_deprecated():
                calls = [call for call in calls if call[0][1] != DeprecationWarning]
            if sys.version_info >= (2, 7, 9) or util.IS_PYOPENSSL or util.IS_SECURETRANSPORT:
                category = calls[0][0][1]
            elif util.HAS_SNI:
                category = calls[1][0][1]
            else:
                category = calls[2][0][1]
            assert category == InsecureRequestWarning