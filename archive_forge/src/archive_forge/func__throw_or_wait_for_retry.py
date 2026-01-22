from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import select
import socket
import ssl
import struct
import threading
from googlecloudsdk.core.util import platforms
import six
import websocket._abnf as websocket_frame_utils
import websocket._exceptions as websocket_exceptions
import websocket._handshake as websocket_handshake
import websocket._http as websocket_http_utils
import websocket._utils as websocket_utils
def _throw_or_wait_for_retry(self, attempt, exception):
    """Wait for the websocket to be ready we don't retry too much too quick."""
    self._throw_on_non_retriable_exception(exception)
    if attempt < WEBSOCKET_MAX_ATTEMPTS and self.sock and (self.sock.fileno() != -1):
        self._wait_for_socket_to_ready(WEBSOCKET_RETRY_TIMEOUT_SECS)
    else:
        raise exception