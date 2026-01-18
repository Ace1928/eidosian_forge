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
@pytest.mark.skipif(sys.version_info < (3, 8), reason='requires python 3.8+')
def test_sslkeylogfile(self, tmpdir, monkeypatch):
    if not hasattr(util.SSLContext, 'keylog_filename'):
        pytest.skip('requires OpenSSL 1.1.1+')
    keylog_file = tmpdir.join('keylogfile.txt')
    monkeypatch.setenv('SSLKEYLOGFILE', str(keylog_file))
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as https_pool:
        r = https_pool.request('GET', '/')
        assert r.status == 200, r.data
        assert keylog_file.check(file=1), "keylogfile '%s' should exist" % str(keylog_file)
        assert keylog_file.read().startswith('# TLS secrets log file'), "keylogfile '%s' should start with '# TLS secrets log file'" % str(keylog_file)