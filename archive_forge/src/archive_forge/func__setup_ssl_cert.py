import logging
import os
import os.path
import socket
import sys
import warnings
from base64 import b64encode
from urllib3 import PoolManager, Timeout, proxy_from_url
from urllib3.exceptions import (
from urllib3.exceptions import (
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import (
from urllib3.util.url import parse_url
import botocore.awsrequest
from botocore.compat import (
from botocore.exceptions import (
def _setup_ssl_cert(self, conn, url, verify):
    if url.lower().startswith('https') and verify:
        conn.cert_reqs = 'CERT_REQUIRED'
        conn.ca_certs = get_cert_path(verify)
    else:
        conn.cert_reqs = 'CERT_NONE'
        conn.ca_certs = None