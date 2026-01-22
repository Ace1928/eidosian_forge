import errno
import os
import re
import socket
import ssl
from contextlib import contextmanager
from ssl import SSLError
from struct import pack, unpack
from .exceptions import UnexpectedFrame
from .platform import KNOWN_TCP_OPTS, SOL_TCP
from .utils import set_cloexec
class _AbstractTransport:
    """Common superclass for TCP and SSL transports.

    PARAMETERS:
        host: str

            Broker address in format ``HOSTNAME:PORT``.

        connect_timeout: int

            Timeout of creating new connection.

        read_timeout: int

            sets ``SO_RCVTIMEO`` parameter of socket.

        write_timeout: int

            sets ``SO_SNDTIMEO`` parameter of socket.

        socket_settings: dict

            dictionary containing `optname` and ``optval`` passed to
            ``setsockopt(2)``.

        raise_on_initial_eintr: bool

            when True, ``socket.timeout`` is raised
            when exception is received during first read. See ``_read()`` for
            details.
    """

    def __init__(self, host, connect_timeout=None, read_timeout=None, write_timeout=None, socket_settings=None, raise_on_initial_eintr=True, **kwargs):
        self.connected = False
        self.sock = None
        self.raise_on_initial_eintr = raise_on_initial_eintr
        self._read_buffer = EMPTY_BUFFER
        self.host, self.port = to_host_port(host)
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        self.socket_settings = socket_settings
    __slots__ = ('connection', 'sock', 'raise_on_initial_eintr', '_read_buffer', 'host', 'port', 'connect_timeout', 'read_timeout', 'write_timeout', 'socket_settings', '__dict__', '__weakref__')

    def __repr__(self):
        if self.sock:
            src = f'{self.sock.getsockname()[0]}:{self.sock.getsockname()[1]}'
            dst = f'{self.sock.getpeername()[0]}:{self.sock.getpeername()[1]}'
            return f'<{type(self).__name__}: {src} -> {dst} at {id(self):#x}>'
        else:
            return f'<{type(self).__name__}: (disconnected) at {id(self):#x}>'

    def connect(self):
        try:
            if self.connected:
                return
            self._connect(self.host, self.port, self.connect_timeout)
            self._init_socket(self.socket_settings, self.read_timeout, self.write_timeout)
            self.connected = True
        except (OSError, SSLError):
            if self.sock and (not self.connected):
                self.sock.close()
                self.sock = None
            raise

    @contextmanager
    def having_timeout(self, timeout):
        if timeout is None:
            yield self.sock
        else:
            sock = self.sock
            prev = sock.gettimeout()
            if prev != timeout:
                sock.settimeout(timeout)
            try:
                yield self.sock
            except SSLError as exc:
                if 'timed out' in str(exc):
                    raise socket.timeout()
                elif 'The operation did not complete' in str(exc):
                    raise socket.timeout()
                raise
            except OSError as exc:
                if exc.errno == errno.EWOULDBLOCK:
                    raise socket.timeout()
                raise
            finally:
                if timeout != prev:
                    sock.settimeout(prev)

    def _connect(self, host, port, timeout):
        entries = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, SOL_TCP)
        for i, res in enumerate(entries):
            af, socktype, proto, canonname, sa = res
            try:
                self.sock = socket.socket(af, socktype, proto)
                try:
                    set_cloexec(self.sock, True)
                except NotImplementedError:
                    pass
                self.sock.settimeout(timeout)
                self.sock.connect(sa)
            except socket.error:
                if self.sock:
                    self.sock.close()
                self.sock = None
                if i + 1 >= len(entries):
                    raise
            else:
                break

    def _init_socket(self, socket_settings, read_timeout, write_timeout):
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self._set_socket_options(socket_settings)
        for timeout, interval in ((socket.SO_SNDTIMEO, write_timeout), (socket.SO_RCVTIMEO, read_timeout)):
            if interval is not None:
                sec = int(interval)
                usec = int((interval - sec) * 1000000)
                self.sock.setsockopt(socket.SOL_SOCKET, timeout, pack('ll', sec, usec))
        self._setup_transport()
        self._write(AMQP_PROTOCOL_HEADER)

    def _get_tcp_socket_defaults(self, sock):
        tcp_opts = {}
        for opt in KNOWN_TCP_OPTS:
            enum = None
            if opt == 'TCP_USER_TIMEOUT':
                try:
                    from socket import TCP_USER_TIMEOUT as enum
                except ImportError:
                    enum = 18
            elif hasattr(socket, opt):
                enum = getattr(socket, opt)
            if enum:
                if opt in DEFAULT_SOCKET_SETTINGS:
                    tcp_opts[enum] = DEFAULT_SOCKET_SETTINGS[opt]
                elif hasattr(socket, opt):
                    tcp_opts[enum] = sock.getsockopt(SOL_TCP, getattr(socket, opt))
        return tcp_opts

    def _set_socket_options(self, socket_settings):
        tcp_opts = self._get_tcp_socket_defaults(self.sock)
        if socket_settings:
            tcp_opts.update(socket_settings)
        for opt, val in tcp_opts.items():
            self.sock.setsockopt(SOL_TCP, opt, val)

    def _read(self, n, initial=False):
        """Read exactly n bytes from the peer."""
        raise NotImplementedError('Must be overridden in subclass')

    def _setup_transport(self):
        """Do any additional initialization of the class."""
        pass

    def _shutdown_transport(self):
        """Do any preliminary work in shutting down the connection."""
        pass

    def _write(self, s):
        """Completely write a string to the peer."""
        raise NotImplementedError('Must be overridden in subclass')

    def close(self):
        if self.sock is not None:
            try:
                self._shutdown_transport()
            except OSError:
                pass
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.connected = False

    def read_frame(self, unpack=unpack):
        """Parse AMQP frame.

        Frame has following format::

            0      1         3         7                   size+7      size+8
            +------+---------+---------+   +-------------+   +-----------+
            | type | channel |  size   |   |   payload   |   | frame-end |
            +------+---------+---------+   +-------------+   +-----------+
             octet    short     long        'size' octets        octet

        """
        read = self._read
        read_frame_buffer = EMPTY_BUFFER
        try:
            frame_header = read(7, True)
            read_frame_buffer += frame_header
            frame_type, channel, size = unpack('>BHI', frame_header)
            if size > SIGNED_INT_MAX:
                part1 = read(SIGNED_INT_MAX)
                try:
                    part2 = read(size - SIGNED_INT_MAX)
                except (socket.timeout, OSError, SSLError):
                    read_frame_buffer += part1
                    raise
                payload = b''.join([part1, part2])
            else:
                payload = read(size)
            read_frame_buffer += payload
            frame_end = ord(read(1))
        except socket.timeout:
            self._read_buffer = read_frame_buffer + self._read_buffer
            raise
        except (OSError, SSLError) as exc:
            if isinstance(exc, socket.error) and os.name == 'nt' and (exc.errno == errno.EWOULDBLOCK):
                self._read_buffer = read_frame_buffer + self._read_buffer
                raise socket.timeout()
            if isinstance(exc, SSLError) and 'timed out' in str(exc):
                self._read_buffer = read_frame_buffer + self._read_buffer
                raise socket.timeout()
            if exc.errno not in _UNAVAIL:
                self.connected = False
            raise
        if frame_end == 206:
            return (frame_type, channel, payload)
        else:
            raise UnexpectedFrame(f'Received frame_end {frame_end:#04x} while expecting 0xce')

    def write(self, s):
        try:
            self._write(s)
        except socket.timeout:
            raise
        except OSError as exc:
            if exc.errno not in _UNAVAIL:
                self.connected = False
            raise