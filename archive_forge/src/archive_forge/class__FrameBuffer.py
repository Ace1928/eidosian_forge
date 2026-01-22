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
class _FrameBuffer(object):
    """Class that represents one single frame sent or received by the websocket."""

    def __init__(self, recv_fn):
        self.recv = recv_fn

    def _recv_header(self):
        """Parse the header from the message."""
        header = self.recv(2)
        b1 = header[0]
        if six.PY2:
            b1 = ord(b1)
        fin = b1 >> 7 & 1
        rsv1 = b1 >> 6 & 1
        rsv2 = b1 >> 5 & 1
        rsv3 = b1 >> 4 & 1
        opcode = b1 & 15
        b2 = header[1]
        if six.PY2:
            b2 = ord(b2)
        has_mask = b2 >> 7 & 1
        length_bits = b2 & 127
        return (fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits)

    def _recv_length(self, bits):
        """Parse the length from the message."""
        length_bits = bits & 127
        if length_bits == 126:
            v = self.recv(2)
            return struct.unpack('!H', v)[0]
        elif length_bits == 127:
            v = self.recv(8)
            return struct.unpack('!Q', v)[0]
        else:
            return length_bits

    def recv_frame(self):
        """Receives the whole frame."""
        fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits = self._recv_header()
        if has_mask == 1:
            raise Exception('Server should not mask the response')
        length = self._recv_length(length_bits)
        payload = self.recv(length)
        return websocket_frame_utils.ABNF(fin, rsv1, rsv2, rsv3, opcode, has_mask, payload)