from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ctypes
import errno
import functools
import gc
import io
import os
import select
import socket
import sys
import threading
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket
from googlecloudsdk.api_lib.compute import iap_tunnel_websocket_utils as utils
from googlecloudsdk.api_lib.compute import sg_tunnel
from googlecloudsdk.api_lib.compute import sg_tunnel_utils as sg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import http_proxy
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import portpicker
import six
from six.moves import queue
def _GetTunnelTargetInfo(self):
    proxy_info = http_proxy.GetHttpProxyInfo()
    if callable(proxy_info):
        proxy_info = proxy_info(method='https')
    return utils.IapTunnelTargetInfo(project=self._project, zone=self._zone, instance=self._instance, interface=self._interface, port=self._port, url_override=self._iap_tunnel_url_override, proxy_info=proxy_info, region=self._region, network=self._network, host=self._host, dest_group=self._dest_group)