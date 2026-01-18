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
def test_warning_for_certs_without_a_san(self, no_san_server):
    """Ensure that a warning is raised when the cert from the server has
        no Subject Alternative Name."""
    with mock.patch('warnings.warn') as warn:
        with HTTPSConnectionPool(no_san_server.host, no_san_server.port, cert_reqs='CERT_REQUIRED', ca_certs=no_san_server.ca_certs) as https_pool:
            r = https_pool.request('GET', '/')
            assert r.status == 200
            assert warn.called