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
def GetNewHttp(http_class=httplib2.Http, **kwargs):
    """Creates and returns a new httplib2.Http instance.

  Args:
    http_class: Optional custom Http class to use.
    **kwargs: Arguments to pass to http_class constructor.

  Returns:
    An initialized httplib2.Http instance.
  """
    proxy_host = config.get('Boto', 'proxy', None)
    boto_proxy_config = {'proxy_host': proxy_host, 'proxy_type': config.get('Boto', 'proxy_type', 'http'), 'proxy_port': config.getint('Boto', 'proxy_port'), 'proxy_user': config.get('Boto', 'proxy_user', None), 'proxy_pass': config.get('Boto', 'proxy_pass', None), 'proxy_rdns': config.get('Boto', 'proxy_rdns', True if proxy_host else None)}
    proxy_info = SetProxyInfo(boto_proxy_config)
    kwargs['ca_certs'] = GetCertsFile()
    kwargs['timeout'] = SSL_TIMEOUT_SEC
    http = http_class(proxy_info=proxy_info, **kwargs)
    http.disable_ssl_certificate_validation = not config.getbool('Boto', 'https_validate_certificates')
    global_context_config = context_config.get_context_config()
    if global_context_config and global_context_config.use_client_certificate:
        http.add_certificate(key=global_context_config.client_cert_path, cert=global_context_config.client_cert_path, domain='', password=global_context_config.client_cert_password)
    return http