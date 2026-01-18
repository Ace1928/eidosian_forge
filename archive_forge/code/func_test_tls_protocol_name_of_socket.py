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
def test_tls_protocol_name_of_socket(self):
    if self.tls_protocol_name is None:
        pytest.skip('Skipping base test class')
    with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as https_pool:
        conn = https_pool._get_conn()
        try:
            conn.connect()
            if not hasattr(conn.sock, 'version'):
                pytest.skip('SSLSocket.version() not available')
            assert conn.sock.version() == self.tls_protocol_name
        finally:
            conn.close()