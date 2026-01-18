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
def test_wrap_socket_failure_resource_leak(self):
    with HTTPSConnectionPool(self.host, self.port, cert_reqs='CERT_REQUIRED', ca_certs=self.bad_ca_path) as https_pool:
        conn = https_pool._get_conn()
        try:
            with pytest.raises(ssl.SSLError):
                conn.connect()
            assert conn.sock
        finally:
            conn.close()