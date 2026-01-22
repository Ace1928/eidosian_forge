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
def _ReadFromStdinAndEnqueueMessageWindows(self):
    """Reads data from stdin on Windows.

      This method will loop until stdin is closed. Should be executed in a
      separate thread to avoid blocking the main thread.
    """
    try:
        while not self._stdin_closed:
            h = ctypes.windll.kernel32.GetStdHandle(-10)
            buf = ctypes.create_string_buffer(self._bufsize)
            number_of_bytes_read = wintypes.DWORD()
            ok = ctypes.windll.kernel32.ReadFile(h, buf, self._bufsize, ctypes.byref(number_of_bytes_read), None)
            if not ok:
                raise socket.error(errno.EIO, 'stdin ReadFile failed')
            msg = buf.raw[:number_of_bytes_read.value]
            self._message_queue.put(self._StdinSocketMessage(self._DataMessageType, msg))
    except Exception:
        self._message_queue.put(self._StdinSocketMessage(self._ExceptionMessageType, sys.exc_info()))