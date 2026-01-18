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
@onlyPy279OrNewer
def test_ssl_wrong_system_time(self):
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as https_pool:
        https_pool.cert_reqs = 'CERT_REQUIRED'
        https_pool.ca_certs = DEFAULT_CA
        with mock.patch('urllib3.connection.datetime') as mock_date:
            mock_date.date.today.return_value = datetime.date(1970, 1, 1)
            w = self._request_without_resource_warnings('GET', '/')
            assert len(w) == 1
            warning = w[0]
            assert SystemTimeWarning == warning.category
            assert str(RECENT_DATE) in warning.message.args[0]