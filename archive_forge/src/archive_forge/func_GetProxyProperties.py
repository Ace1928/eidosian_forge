from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
from six.moves import urllib
def GetProxyProperties():
    """Get proxy information from cloud sdk properties in dictionary form."""
    proxy_type_map = http_proxy_types.PROXY_TYPE_MAP
    proxy_type = properties.VALUES.proxy.proxy_type.Get()
    proxy_address = properties.VALUES.proxy.address.Get()
    proxy_port = properties.VALUES.proxy.port.GetInt()
    proxy_prop_set = len([f for f in (proxy_type, proxy_address, proxy_port) if f])
    if proxy_prop_set > 0 and proxy_prop_set != 3:
        raise properties.InvalidValueError('Please set all or none of the following properties: proxy/type, proxy/address and proxy/port')
    if not proxy_prop_set:
        return {}
    proxy_rdns = properties.VALUES.proxy.rdns.GetBool()
    proxy_user = properties.VALUES.proxy.username.Get()
    proxy_pass = properties.VALUES.proxy.password.Get()
    return {'proxy_type': proxy_type_map[proxy_type], 'proxy_address': proxy_address, 'proxy_port': proxy_port, 'proxy_rdns': proxy_rdns, 'proxy_user': proxy_user, 'proxy_pass': proxy_pass}