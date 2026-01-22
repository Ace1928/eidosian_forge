from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
import httplib2
def EffectiveProxyInfo():
    """Returns ProxyInfo effective in gcloud and if it is from gloud properties.

  Returns:
    A tuple of two elements in which the first element is an httplib2.ProxyInfo
      object and the second is a bool that is True if the proxy info came from
      previously set Cloud SDK proxy properties.

  Raises:
    properties.InvalidValueError: If the properties did not include a valid set.
      "Valid" means all three of these attributes are present: proxy type, host,
      and port.
  """
    proxy_info = http_proxy.GetHttpProxyInfo()
    if not proxy_info:
        return (None, False)
    from_gcloud_properties = True
    if not isinstance(proxy_info, httplib2.ProxyInfo):
        from_gcloud_properties = False
        proxy_info = proxy_info('https')
    return (proxy_info, from_gcloud_properties)