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
def _InitiateConnection(self, local_conn, get_access_token_callback, user_agent, conn_id=-1):
    tunnel_target = self._GetTunnelTargetInfo()
    new_websocket = iap_tunnel_websocket.IapTunnelWebSocket(tunnel_target, get_access_token_callback, functools.partial(_SendLocalDataCallback, local_conn), functools.partial(_CloseLocalConnectionCallback, local_conn), user_agent, ignore_certs=self._ignore_certs, conn_id=conn_id)
    new_websocket.InitiateConnection()
    return new_websocket