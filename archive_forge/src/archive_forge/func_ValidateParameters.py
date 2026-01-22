from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import struct
import sys
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import http_proxy_types
import httplib2
import six
from six.moves.urllib import parse
import socks
def ValidateParameters(tunnel_target):
    """Validate the parameters.

  Inspects the parameters to ensure that they are valid for either a VM
  instance-based connection, or a host-based connection.

  Args:
    tunnel_target: The argument container.

  Raises:
    MissingTunnelParameter: A required argument is missing.
    UnexpectedTunnelParameter: An unexpected argument was found.
    UnsupportedProxyType: A non-http proxy was specified.
  """
    for field_name, field_value in tunnel_target._asdict().items():
        if not field_value and field_name in ('project', 'port'):
            raise MissingTunnelParameter('Missing required tunnel argument: ' + field_name)
    if tunnel_target.region or tunnel_target.network or tunnel_target.host or tunnel_target.dest_group:
        for field_name, field_value in tunnel_target._asdict().items():
            if not field_value and field_name in ('region', 'network', 'host'):
                raise MissingTunnelParameter('Missing required tunnel argument: ' + field_name)
            if field_value and field_name in ('instance', 'interface', 'zone'):
                raise UnexpectedTunnelParameter('Unexpected tunnel argument: ' + field_name)
    else:
        for field_name, field_value in tunnel_target._asdict().items():
            if not field_value and field_name in ('zone', 'instance', 'interface'):
                raise MissingTunnelParameter('Missing required tunnel argument: ' + field_name)
    if tunnel_target.proxy_info:
        proxy_type = tunnel_target.proxy_info.proxy_type
        if proxy_type and proxy_type != socks.PROXY_TYPE_HTTP:
            raise UnsupportedProxyType('Unsupported proxy type: ' + http_proxy_types.REVERSE_PROXY_TYPE_MAP[proxy_type])