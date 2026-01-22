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
class SockOpt(object):
    """Class that represents the options for the underlying socket library."""

    def __init__(self, sslopt):
        if sslopt is None:
            sslopt = {}
        self.timeout = None
        self.sockopt = []
        self.sslopt = sslopt