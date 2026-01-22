import errno
import os.path
import socket
import sys
import threading
import time
from ... import errors, trace
from ... import transport as _mod_transport
from ...hooks import Hooks
from ...i18n import gettext
from ...lazy_import import lazy_import
from breezy.bzr.smart import (
from breezy.transport import (
from breezy import (
class SmartTCPServer:
    """Listens on a TCP socket and accepts connections from smart clients.

    Each connection will be served by a SmartServerSocketStreamMedium running in
    a thread.

    hooks: An instance of SmartServerHooks.
    """
    _ACCEPT_TIMEOUT = 1.0
    _SHUTDOWN_POLL_TIMEOUT = 1.0
    _LOG_WAITING_TIMEOUT = 10.0
    _timer = time.time

    def __init__(self, backing_transport, root_client_path='/', client_timeout=None):
        """Construct a new server.

        To actually start it running, call either start_background_thread or
        serve.

        :param backing_transport: The transport to serve.
        :param root_client_path: The client path that will correspond to root
            of backing_transport.
        :param client_timeout: See SmartServerSocketStreamMedium's timeout
            parameter.
        """
        self.backing_transport = backing_transport
        self.root_client_path = root_client_path
        self._client_timeout = client_timeout
        self._active_connections = []
        self._gracefully_stopping = False

    def start_server(self, host, port):
        """Create the server listening socket.

        :param host: Name of the interface to listen on.
        :param port: TCP port to listen on, or 0 to allocate a transient port.
        """
        from socket import error as socket_error
        from socket import timeout as socket_timeout
        self._socket_error = socket_error
        self._socket_timeout = socket_timeout
        addrs = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)[0]
        family, socktype, proto, canonname, sockaddr = addrs
        self._server_socket = socket.socket(family, socktype, proto)
        if sys.platform != 'win32':
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._server_socket.bind(sockaddr)
        except self._socket_error as message:
            raise errors.CannotBindAddress(host, port, message)
        self._sockname = self._server_socket.getsockname()
        self.port = self._sockname[1]
        self._server_socket.listen(1)
        self._server_socket.settimeout(self._ACCEPT_TIMEOUT)
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._fully_stopped = threading.Event()

    def _backing_urls(self):
        urls = [self.backing_transport.base]
        try:
            urls.append(self.backing_transport.external_url())
        except errors.InProcessTransport:
            pass
        return urls

    def run_server_started_hooks(self, backing_urls=None):
        if backing_urls is None:
            backing_urls = self._backing_urls()
        for hook in SmartTCPServer.hooks['server_started']:
            hook(backing_urls, self.get_url())
        for hook in SmartTCPServer.hooks['server_started_ex']:
            hook(backing_urls, self)

    def run_server_stopped_hooks(self, backing_urls=None):
        if backing_urls is None:
            backing_urls = self._backing_urls()
        for hook in SmartTCPServer.hooks['server_stopped']:
            hook(backing_urls, self.get_url())

    def _stop_gracefully(self):
        trace.note(gettext('Requested to stop gracefully'))
        self._should_terminate = True
        self._gracefully_stopping = True
        for handler, _ in self._active_connections:
            handler._stop_gracefully()

    def _wait_for_clients_to_disconnect(self):
        self._poll_active_connections()
        if not self._active_connections:
            return
        trace.note(gettext('Waiting for %d client(s) to finish') % (len(self._active_connections),))
        t_next_log = self._timer() + self._LOG_WAITING_TIMEOUT
        while self._active_connections:
            now = self._timer()
            if now >= t_next_log:
                trace.note(gettext('Still waiting for %d client(s) to finish') % (len(self._active_connections),))
                t_next_log = now + self._LOG_WAITING_TIMEOUT
            self._poll_active_connections(self._SHUTDOWN_POLL_TIMEOUT)

    def serve(self, thread_name_suffix=''):
        stop_gracefully = self._stop_gracefully
        signals.register_on_hangup(id(self), stop_gracefully)
        self._should_terminate = False
        self.run_server_started_hooks()
        self._started.set()
        try:
            try:
                while not self._should_terminate:
                    try:
                        conn, client_addr = self._server_socket.accept()
                    except self._socket_timeout:
                        pass
                    except self._socket_error as e:
                        if e.args[0] not in (errno.EBADF, errno.EINTR):
                            trace.warning(gettext('listening socket error: %s') % (e,))
                    else:
                        if self._should_terminate:
                            conn.close()
                            break
                        self.serve_conn(conn, thread_name_suffix)
                    self._poll_active_connections()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                trace.report_exception(sys.exc_info(), sys.stderr)
                raise
        finally:
            try:
                self._server_socket.close()
            except self._socket_error:
                pass
            self._stopped.set()
            signals.unregister_on_hangup(id(self))
            self.run_server_stopped_hooks()
        if self._gracefully_stopping:
            self._wait_for_clients_to_disconnect()
        self._fully_stopped.set()

    def get_url(self):
        """Return the url of the server"""
        return 'bzr://{}:{}/'.format(self._sockname[0], self._sockname[1])

    def _make_handler(self, conn):
        return medium.SmartServerSocketStreamMedium(conn, self.backing_transport, self.root_client_path, timeout=self._client_timeout)

    def _poll_active_connections(self, timeout=0.0):
        """Check to see if any active connections have finished.

        This will iterate through self._active_connections, and update any
        connections that are finished.

        :param timeout: The timeout to pass to thread.join(). By default, we
            set it to 0, so that we don't hang if threads are not done yet.
        :return: None
        """
        still_active = []
        for handler, thread in self._active_connections:
            thread.join(timeout)
            if thread.is_alive():
                still_active.append((handler, thread))
        self._active_connections = still_active

    def serve_conn(self, conn, thread_name_suffix):
        conn.setblocking(True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        thread_name = 'smart-server-child' + thread_name_suffix
        handler = self._make_handler(conn)
        connection_thread = threading.Thread(None, handler.serve, name=thread_name, daemon=True)
        self._active_connections.append((handler, connection_thread))
        connection_thread.start()
        return connection_thread

    def start_background_thread(self, thread_name_suffix=''):
        self._started.clear()
        self._server_thread = threading.Thread(None, self.serve, args=(thread_name_suffix,), name='server-' + self.get_url(), daemon=True)
        self._server_thread.start()
        self._started.wait()

    def stop_background_thread(self):
        self._stopped.clear()
        self._should_terminate = True
        try:
            self._server_socket.close()
        except self._socket_error:
            pass
        if not self._stopped.is_set():
            temp_socket = socket.socket()
            temp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if not temp_socket.connect_ex(self._sockname):
                temp_socket.close()
        self._stopped.wait()
        self._server_thread.join()