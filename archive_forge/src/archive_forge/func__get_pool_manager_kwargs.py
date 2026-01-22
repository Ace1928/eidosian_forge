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
def _get_pool_manager_kwargs(self, **extra_kwargs):
    pool_manager_kwargs = {'timeout': self._timeout, 'maxsize': self._max_pool_connections, 'ssl_context': self._get_ssl_context(), 'socket_options': self._socket_options, 'cert_file': self._cert_file, 'key_file': self._key_file}
    pool_manager_kwargs.update(**extra_kwargs)
    return pool_manager_kwargs