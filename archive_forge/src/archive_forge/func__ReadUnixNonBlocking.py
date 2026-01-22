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
def _ReadUnixNonBlocking(self, bufsize):
    """Reads from stdin on Unix in a nonblocking manner.

    Args:
      bufsize: The maximum number of bytes to receive. Must be positive.
    Returns:
      The bytes read. b'' means no data is available.
    Raises:
      _StdinSocket._EOFError: to indicate EOF.
      IOError: On low level errors.
    """
    try:
        if six.PY2:
            b = sys.stdin.read(bufsize)
        else:
            b = sys.stdin.buffer.read(bufsize)
    except IOError as e:
        if e.errno == errno.EAGAIN or isinstance(e, io.BlockingIOError):
            return b''
        raise
    if b == b'':
        raise _StdinSocket._EOFError
    if b is None:
        b = b''
    return b