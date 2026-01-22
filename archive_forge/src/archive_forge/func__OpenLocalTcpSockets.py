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
def _OpenLocalTcpSockets(local_host, local_port):
    """Attempt to open a local socket(s) listening on specified host and port."""
    open_sockets = []
    for res in socket.getaddrinfo(local_host, local_port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        af, socktype, proto, unused_canonname, sock_addr = res
        try:
            s = socket.socket(af, socktype, proto)
        except socket.error:
            continue
        try:
            if not platforms.OperatingSystem.IsWindows():
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(sock_addr)
            s.listen(128)
            open_sockets.append(s)
        except EnvironmentError:
            try:
                s.close()
            except socket.error:
                pass
    if open_sockets:
        return open_sockets
    raise UnableToOpenPortError('Unable to open socket on port [%d].' % local_port)