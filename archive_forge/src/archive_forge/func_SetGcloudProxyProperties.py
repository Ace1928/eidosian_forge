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
def SetGcloudProxyProperties(proxy_type=None, address=None, port=None, username=None, password=None):
    """Sets proxy group properties; clears any property not explicitly set."""
    properties.PersistProperty(properties.VALUES.proxy.proxy_type, proxy_type)
    properties.PersistProperty(properties.VALUES.proxy.address, address)
    properties.PersistProperty(properties.VALUES.proxy.port, port)
    properties.PersistProperty(properties.VALUES.proxy.username, username)
    properties.PersistProperty(properties.VALUES.proxy.password, password)