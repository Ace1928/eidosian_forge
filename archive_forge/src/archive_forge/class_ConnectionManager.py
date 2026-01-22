import io
import os
import socket
import threading
import time
import selectors
from contextlib import suppress
from . import errors
from ._compat import IS_WINDOWS
from .makefile import MakeFile
class ConnectionManager:
    """Class which manages HTTPConnection objects.

    This is for connections which are being kept-alive for follow-up requests.
    """

    def __init__(self, server):
        """Initialize ConnectionManager object.

        Args:
            server (cheroot.server.HTTPServer): web server object
                that uses this ConnectionManager instance.
        """
        self._serving = False
        self._stop_requested = False
        self.server = server
        self._selector = _ThreadsafeSelector()
        self._selector.register(server.socket.fileno(), selectors.EVENT_READ, data=server)

    def put(self, conn):
        """Put idle connection into the ConnectionManager to be managed.

        :param conn: HTTP connection to be managed
        :type conn: cheroot.server.HTTPConnection
        """
        conn.last_used = time.time()
        if conn.rfile.has_data():
            self.server.process_conn(conn)
        else:
            self._selector.register(conn.socket.fileno(), selectors.EVENT_READ, data=conn)

    def _expire(self, threshold):
        """Expire least recently used connections.

        :param threshold: Connections that have not been used within this \\
                          duration (in seconds), are considered expired and \\
                          are closed and removed.
        :type threshold: float

        This should be called periodically.
        """
        timed_out_connections = [(sock_fd, conn) for sock_fd, conn in self._selector.connections if conn != self.server and conn.last_used < threshold]
        for sock_fd, conn in timed_out_connections:
            self._selector.unregister(sock_fd)
            conn.close()

    def stop(self):
        """Stop the selector loop in run() synchronously.

        May take up to half a second.
        """
        self._stop_requested = True
        while self._serving:
            time.sleep(0.01)

    def run(self, expiration_interval):
        """Run the connections selector indefinitely.

        Args:
            expiration_interval (float): Interval, in seconds, at which
                connections will be checked for expiration.

        Connections that are ready to process are submitted via
        self.server.process_conn()

        Connections submitted for processing must be `put()`
        back if they should be examined again for another request.

        Can be shut down by calling `stop()`.
        """
        self._serving = True
        try:
            self._run(expiration_interval)
        finally:
            self._serving = False

    def _run(self, expiration_interval):
        """Run connection handler loop until stop was requested.

        :param expiration_interval: Interval, in seconds, at which \\
                                    connections will be checked for \\
                                    expiration.
        :type expiration_interval: float

        Use ``expiration_interval`` as ``select()`` timeout
        to assure expired connections are closed in time.

        On Windows cap the timeout to 0.05 seconds
        as ``select()`` does not return when a socket is ready.
        """
        last_expiration_check = time.time()
        if IS_WINDOWS:
            select_timeout = min(expiration_interval, 0.05)
        else:
            select_timeout = expiration_interval
        while not self._stop_requested:
            try:
                active_list = self._selector.select(timeout=select_timeout)
            except OSError:
                self._remove_invalid_sockets()
                continue
            for sock_fd, conn in active_list:
                if conn is self.server:
                    new_conn = self._from_server_socket(self.server.socket)
                    if new_conn is not None:
                        self.server.process_conn(new_conn)
                else:
                    self._selector.unregister(sock_fd)
                    self.server.process_conn(conn)
            now = time.time()
            if now - last_expiration_check > expiration_interval:
                self._expire(threshold=now - self.server.timeout)
                last_expiration_check = now

    def _remove_invalid_sockets(self):
        """Clean up the resources of any broken connections.

        This method attempts to detect any connections in an invalid state,
        unregisters them from the selector and closes the file descriptors of
        the corresponding network sockets where possible.
        """
        invalid_conns = []
        for sock_fd, conn in self._selector.connections:
            if conn is self.server:
                continue
            try:
                os.fstat(sock_fd)
            except OSError:
                invalid_conns.append((sock_fd, conn))
        for sock_fd, conn in invalid_conns:
            self._selector.unregister(sock_fd)
            with suppress(OSError):
                conn.close()

    def _from_server_socket(self, server_socket):
        try:
            s, addr = server_socket.accept()
            if self.server.stats['Enabled']:
                self.server.stats['Accepts'] += 1
            prevent_socket_inheritance(s)
            if hasattr(s, 'settimeout'):
                s.settimeout(self.server.timeout)
            mf = MakeFile
            ssl_env = {}
            if self.server.ssl_adapter is not None:
                try:
                    s, ssl_env = self.server.ssl_adapter.wrap(s)
                except errors.NoSSLError:
                    msg = 'The client sent a plain HTTP request, but this server only speaks HTTPS on this port.'
                    buf = ['%s 400 Bad Request\r\n' % self.server.protocol, 'Content-Length: %s\r\n' % len(msg), 'Content-Type: text/plain\r\n\r\n', msg]
                    wfile = mf(s, 'wb', io.DEFAULT_BUFFER_SIZE)
                    try:
                        wfile.write(''.join(buf).encode('ISO-8859-1'))
                    except OSError as ex:
                        if ex.args[0] not in errors.socket_errors_to_ignore:
                            raise
                    return
                if not s:
                    return
                mf = self.server.ssl_adapter.makefile
                if hasattr(s, 'settimeout'):
                    s.settimeout(self.server.timeout)
            conn = self.server.ConnectionClass(self.server, s, mf)
            if not isinstance(self.server.bind_addr, (str, bytes)):
                if addr is None:
                    if len(s.getsockname()) == 2:
                        addr = ('0.0.0.0', 0)
                    else:
                        addr = ('::', 0)
                conn.remote_addr = addr[0]
                conn.remote_port = addr[1]
            conn.ssl_env = ssl_env
            return conn
        except socket.timeout:
            return
        except OSError as ex:
            if self.server.stats['Enabled']:
                self.server.stats['Socket Errors'] += 1
            if ex.args[0] in errors.socket_error_eintr:
                return
            if ex.args[0] in errors.socket_errors_nonblocking:
                return
            if ex.args[0] in errors.socket_errors_to_ignore:
                return
            raise

    def close(self):
        """Close all monitored connections."""
        for _, conn in self._selector.connections:
            if conn is not self.server:
                conn.close()
        self._selector.close()

    @property
    def _num_connections(self):
        """Return the current number of connections.

        Includes all connections registered with the selector,
        minus one for the server socket, which is always registered
        with the selector.
        """
        return len(self._selector) - 1

    @property
    def can_add_keepalive_connection(self):
        """Flag whether it is allowed to add a new keep-alive connection."""
        ka_limit = self.server.keep_alive_conn_limit
        return ka_limit is None or self._num_connections < ka_limit