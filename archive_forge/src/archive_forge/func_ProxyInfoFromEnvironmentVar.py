from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
import tempfile
import textwrap
import six
import boto
from boto import config
import boto.auth
from boto.exception import NoAuthHandlerFound
from boto.gs.connection import GSConnection
from boto.provider import Provider
from boto.pyami.config import BotoConfigLocations
import gslib
from gslib import context_config
from gslib.exception import CommandException
from gslib.utils import system_util
from gslib.utils.constants import DEFAULT_GCS_JSON_API_VERSION
from gslib.utils.constants import DEFAULT_GSUTIL_STATE_DIR
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import UTF8
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import ONE_MIB
import httplib2
from oauth2client.client import HAS_CRYPTO
def ProxyInfoFromEnvironmentVar(proxy_env_var):
    """Reads proxy info from the environment and converts to httplib2.ProxyInfo.

  Args:
    proxy_env_var: Environment variable string to read, such as http_proxy or
       https_proxy.

  Returns:
    httplib2.ProxyInfo constructed from the environment string.
  """
    proxy_url = os.environ.get(proxy_env_var)
    if not proxy_url or not proxy_env_var.lower().startswith('http'):
        return httplib2.ProxyInfo(httplib2.socks.PROXY_TYPE_HTTP, None, 0)
    proxy_protocol = proxy_env_var.lower().split('_')[0]
    if not proxy_url.lower().startswith('http'):
        proxy_url = proxy_protocol + '://' + proxy_url
    return httplib2.proxy_info_from_url(proxy_url, method=proxy_protocol)