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
class _StdinSocket(object):
    """A wrapper around stdin/out that allows it to be treated like a socket.

  Does not implement all socket functions. And of the ones implemented, not all
  arguments/flags are supported. Once created, stdin should never be accessed by
  anything else.
  """

    class _StdinSocketMessage(object):
        """A class to wrap messages coming to the stdin socket for windows systems."""

        def __init__(self, messageType, data):
            self._type = messageType
            self._data = data

        def GetData(self):
            return self._data

        def GetType(self):
            return self._type

    class _EOFError(Exception):
        pass

    class _StdinClosedMessageType:
        pass

    class _ExceptionMessageType:
        pass

    class _DataMessageType:
        pass

    def __init__(self):
        self._stdin_closed = False
        self._bufsize = utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE
        if platforms.OperatingSystem.IsWindows():
            self._message_queue = queue.Queue()
            self._reading_thread = threading.Thread(target=self._ReadFromStdinAndEnqueueMessageWindows)
            self._reading_thread.daemon = True
            self._reading_thread.start()
        else:
            self._old_flags = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)

    def __del__(self):
        if not platforms.OperatingSystem.IsWindows():
            fcntl.fcntl(sys.stdin, fcntl.F_SETFL, self._old_flags)

    def send(self, data):
        files.WriteStreamBytes(sys.stdout, data)
        if not six.PY2:
            sys.stdout.buffer.flush()
        return len(data)

    def recv(self, bufsize):
        """Receives data from stdin.

    Blocks until at least 1 byte is available.
    On Unix (but not Windows) this is unblocked by close() and shutdown(RD).
    On all platforms a signal handler triggering an exception will unblock this.
    This cannot be called by multiple threads at the same time.
    This function performs cleanups before returning, so killing gcloud while
    this is running should be avoided. Specifically RaisesKeyboardInterrupt
    should be in effect so that ctrl-c causes a clean exit with an exception
    instead of triggering gcloud's default os.kill().

    Args:
      bufsize: The maximum number of bytes to receive. Must be positive.
    Returns:
      The bytes received. EOF is indicated by b''.
    Raises:
      IOError: On low level errors.
    """
        if platforms.OperatingSystem.IsWindows():
            return self._RecvWindows(bufsize)
        else:
            return self._RecvUnix(bufsize)

    def close(self):
        self.shutdown(socket.SHUT_RD)

    def shutdown(self, how):
        if how in (socket.SHUT_RDWR, socket.SHUT_RD):
            self._stdin_closed = True
            if platforms.OperatingSystem.IsWindows():
                msg = self._StdinSocketMessage(self._StdinClosedMessageType, b'')
                self._message_queue.put(msg)

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

    def _RecvWindows(self, bufsize):
        """Reads data from stdin on Windows.

    Args:
      bufsize: The maximum number of bytes to receive. Must be positive.
    Returns:
      The bytes received. EOF is indicated by b''.
    Raises:
      socket.error: On low level errors.
    """
        if bufsize != utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE:
            log.info('bufsize [%s] is not max_data_frame_size', bufsize)
        while not self._stdin_closed:
            try:
                msg = self._message_queue.get(timeout=1)
            except queue.Empty:
                continue
            msg_type = msg.GetType()
            msg_data = msg.GetData()
            if msg_type is self._ExceptionMessageType:
                six.reraise(msg_data[0], msg_data[1], msg_data[2])
            if msg_type is self._StdinClosedMessageType:
                self._stdin_closed = True
            return msg_data
        return b''

    def _RecvUnix(self, bufsize):
        """Reads data from stdin on Unix.

    Args:
      bufsize: The maximum number of bytes to receive. Must be positive.
    Returns:
      The bytes received. EOF is indicated by b''. Once EOF has been indicated,
      will always indicate EOF.
    Raises:
      IOError: On low level errors.
    """
        if bufsize != utils.SUBPROTOCOL_MAX_DATA_FRAME_SIZE:
            log.info('bufsize [%s] is not max_data_frame_size', bufsize)
        if self._stdin_closed:
            return b''
        try:
            while not self._stdin_closed:
                stdin_ready = select.select([sys.stdin], (), (), READ_FROM_STDIN_TIMEOUT_SECS)
                if not stdin_ready[0]:
                    continue
                return self._ReadUnixNonBlocking(self._bufsize)
        except _StdinSocket._EOFError:
            self._stdin_closed = True
        return b''

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