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
def test_verified_without_ca_certs(self):
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_REQUIRED') as https_pool:
        with pytest.raises(MaxRetryError) as e:
            https_pool.request('GET', '/')
        assert isinstance(e.value.reason, SSLError)
        assert 'No root certificates specified' in str(e.value.reason) or 'certificate verify failed' in str(e.value.reason).lower() or 'invalid certificate chain' in str(e.value.reason), "Expected 'No root certificates specified',  'certificate verify failed', or 'invalid certificate chain', instead got: %r" % e.value.reason