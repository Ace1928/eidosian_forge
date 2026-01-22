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
def _wait_for_socket_to_ready(self, timeout):
    """Wait for socket to be ready and treat some special errors cases."""
    if self.sock.pending():
        return
    try:
        _ = select.select([self.sock], (), (), timeout)
    except TypeError as e:
        message = websocket_utils.extract_err_message(e)
        if isinstance(message, str) and 'arguments 1-3 must be sequences' in message:
            raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while waiting for socket to ready.')
        raise
    except (OSError, select.error) as e:
        if not platforms.OperatingSystem.IsWindows():
            raise
        if e is OSError and e.winerror != 10038:
            raise
        if e is select.error and e.errno != errno.ENOTSOCK:
            raise