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
def _proxies_kwargs(self, **kwargs):
    proxies_settings = self._proxy_config.settings
    proxies_kwargs = {'use_forwarding_for_https': proxies_settings.get('proxy_use_forwarding_for_https'), **kwargs}
    return {k: v for k, v in proxies_kwargs.items() if v is not None}