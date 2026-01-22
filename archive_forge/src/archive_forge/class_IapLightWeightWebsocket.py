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
class IapLightWeightWebsocket(object):
    """Lightweight websocket created to send and receive data as fast as possible.

     This websocket implements rfc6455
  """

    def __init__(self, url, header, on_data, on_close, on_error, subprotocols, sock=None):
        self.url = url
        self.on_data = on_data
        self.on_close = on_close
        self.on_error = on_error
        self.sock = sock
        self.frame_buffer = _FrameBuffer(self._recv_bytes)
        self.connected = False
        self.get_mask_key = None
        self.subprotocols = subprotocols
        self.header = header
        self.send_write_lock = threading.Lock()

    def recv(self):
        """Receives data from the server."""
        if not self.connected or not self.sock:
            raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while receiving data.')
        return self.frame_buffer.recv_frame()

    def send(self, data, opcode):
        """Sends data to the server."""
        if opcode not in VALID_OPCODES:
            raise ValueError('Invalid opcode')
        frame_data = websocket_frame_utils.ABNF(fin=1, rsv1=0, rsv2=0, rsv3=0, opcode=opcode, mask=1, data=data)
        if self.get_mask_key:
            frame_data.get_mask_key = self.get_mask_key
        frame_data = frame_data.format()
        with self.send_write_lock:
            for attempt in range(1, WEBSOCKET_MAX_ATTEMPTS + 1):
                try:
                    if not self.connected or not self.sock:
                        raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while sending data.')
                    bytes_sent = self.sock.send(frame_data)
                    if not bytes_sent:
                        raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while sending data.')
                    if len(frame_data) != bytes_sent:
                        raise Exception("Packet was not sent in it's entirety")
                    return bytes_sent
                except Exception as e:
                    self._throw_or_wait_for_retry(attempt=attempt, exception=e)

    def close(self, close_code=CLOSE_STATUS_NORMAL, close_message=six.b('')):
        """Closes the connection."""
        if self.connected and self.sock:
            try:
                self.send_close(close_code, close_message)
                self.sock.close()
                self.sock = None
                self.connected = False
            except (websocket_exceptions.WebSocketConnectionClosedException, socket.error) as e:
                if not self._is_closed_connection_exception(e):
                    raise

    def send_close(self, close_code=CLOSE_STATUS_NORMAL, close_message=six.b('')):
        """Sends a close message to the server but don't close."""
        if self.connected:
            if six.PY2:
                close_message = close_message.encode('latin-1')
            try:
                self.send(struct.pack('!H', close_code) + close_message, websocket_frame_utils.ABNF.OPCODE_CLOSE)
            except (websocket_exceptions.WebSocketConnectionClosedException, OSError, socket.error, ssl.SSLError) as e:
                if not self._is_closed_connection_exception(e):
                    raise

    def run_forever(self, sslopt, **options):
        """Main method that will stay running while the connection is open."""
        try:
            options.update({'header': self.header})
            options.update({'subprotocols': self.subprotocols})
            self._connect(sslopt, **options)
            while self.connected:
                if self.sock.fileno() == -1:
                    raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while receiving data.')
                self._wait_for_socket_to_ready(timeout=WEBSOCKET_RETRY_TIMEOUT_SECS)
                frame = self.recv()
                if frame.opcode == websocket_frame_utils.ABNF.OPCODE_CLOSE:
                    close_args = self._get_close_args(frame.data)
                    self.close()
                    self.on_close(*close_args)
                else:
                    self.on_data(frame.data, frame.opcode)
        except KeyboardInterrupt as e:
            self.close()
            self.on_close(close_code=None, close_message=None)
            raise e
        except Exception as e:
            self.close()
            self.on_error(e)
            error_code = websocket_utils.extract_error_code(e)
            message = websocket_utils.extract_err_message(e)
            self.on_close(error_code, message)

    def _recv_bytes(self, buffersize):
        """Internal implementation of recv called by recv_frame."""
        view = memoryview(bytearray(buffersize))
        total_bytes_read = 0
        for attempt in range(1, WEBSOCKET_MAX_ATTEMPTS + 1):
            try:
                while total_bytes_read < buffersize:
                    bytes_received = self.sock.recv_into(view[total_bytes_read:], buffersize - total_bytes_read)
                    if bytes_received == 0:
                        self.close()
                        raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while receiving data.')
                    total_bytes_read += bytes_received
                return view.tobytes()
            except Exception as e:
                self._throw_or_wait_for_retry(attempt=attempt, exception=e)

    def _set_mask_key(self, mask_key):
        self.get_mask_key = mask_key

    def _connect(self, ssl_opt, **options):
        """Connect method, doesn't follow redirects."""
        proxy = websocket_http_utils.proxy_info(**options)
        sockopt = SockOpt(ssl_opt)
        if self.sock:
            hostname, port, resource, _ = websocket_http_utils.parse_url(self.url)
            addrs = (hostname, port, resource)
        else:
            self.sock, addrs = websocket_http_utils.connect(self.url, sockopt, proxy, None)
            websocket_handshake.handshake(self.sock, *addrs, **options)
        self.connected = True
        return addrs

    def _throw_on_non_retriable_exception(self, e):
        """Decides if we throw or if we ignore the exception because it's retriable."""
        if self._is_closed_connection_exception(e):
            raise websocket_exceptions.WebSocketConnectionClosedException('Connection closed while waiting for retry.')
        if e is ssl.SSLError:
            if e.args[0] != ssl.SSL_ERROR_WANT_WRITE:
                raise e
        elif e is socket.error:
            error_code = websocket_utils.extract_error_code(e)
            if error_code is None:
                raise e
            if error_code != errno.EAGAIN or error_code != errno.EWOULDBLOCK:
                raise e

    def _throw_or_wait_for_retry(self, attempt, exception):
        """Wait for the websocket to be ready we don't retry too much too quick."""
        self._throw_on_non_retriable_exception(exception)
        if attempt < WEBSOCKET_MAX_ATTEMPTS and self.sock and (self.sock.fileno() != -1):
            self._wait_for_socket_to_ready(WEBSOCKET_RETRY_TIMEOUT_SECS)
        else:
            raise exception

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

    def _is_closed_connection_exception(self, exception):
        """Method to identify if the exception is of closed connection type."""
        if exception is websocket_exceptions.WebSocketConnectionClosedException:
            return True
        elif exception is OSError and exception.errno == errno.EBADF:
            return True
        elif exception is ssl.SSLError:
            if exception.args[0] == ssl.SSL_ERROR_EOF:
                return True
        else:
            error_code = websocket_utils.extract_error_code(exception)
            if error_code == errno.ENOTCONN or error_code == errno.EPIPE:
                return True
        return False

    def _get_close_args(self, data):
        if data and len(data) >= 2:
            code = 256 * six.byte2int(data[0:1]) + six.byte2int(data[1:2])
            reason = data[2:].decode('utf-8')
            return [code, reason]