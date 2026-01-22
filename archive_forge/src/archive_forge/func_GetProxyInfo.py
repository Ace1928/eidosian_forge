from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import inspect
import io
from google.auth.transport import requests as google_auth_requests
from google.auth.transport.requests import _MutualTlsOffloadAdapter
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import platforms
import httplib2
import requests
import six
from six.moves import http_client as httplib
from six.moves import urllib
import socks
from urllib3.util.ssl_ import create_urllib3_context
def GetProxyInfo():
    """Returns the proxy string for use by requests from gcloud properties.

  See https://requests.readthedocs.io/en/master/user/advanced/#proxies.
  """
    proxy_type = properties.VALUES.proxy.proxy_type.Get()
    proxy_address = properties.VALUES.proxy.address.Get()
    proxy_port = properties.VALUES.proxy.port.GetInt()
    proxy_prop_set = len([f for f in (proxy_type, proxy_address, proxy_port) if f])
    if proxy_prop_set > 0 and proxy_prop_set != 3:
        raise properties.InvalidValueError('Please set all or none of the following properties: proxy/type, proxy/address and proxy/port')
    if not proxy_prop_set:
        return
    proxy_rdns = properties.VALUES.proxy.rdns.GetBool()
    proxy_user = properties.VALUES.proxy.username.Get()
    proxy_pass = properties.VALUES.proxy.password.Get()
    http_proxy_type = http_proxy_types.PROXY_TYPE_MAP[proxy_type]
    if http_proxy_type == socks.PROXY_TYPE_SOCKS4:
        proxy_scheme = 'socks4a' if proxy_rdns else 'socks4'
    elif http_proxy_type == socks.PROXY_TYPE_SOCKS5:
        proxy_scheme = 'socks5h' if proxy_rdns else 'socks5'
    elif http_proxy_type == socks.PROXY_TYPE_HTTP:
        proxy_scheme = 'http'
    else:
        raise ValueError('Unsupported proxy type: {}'.format(proxy_type))
    if proxy_user or proxy_pass:
        proxy_auth = ':'.join((urllib.parse.quote(x) or '' for x in (proxy_user, proxy_pass)))
        proxy_auth += '@'
    else:
        proxy_auth = ''
    return '{}://{}{}:{}'.format(proxy_scheme, proxy_auth, proxy_address, proxy_port)