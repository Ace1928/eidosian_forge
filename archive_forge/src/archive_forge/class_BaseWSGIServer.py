from __future__ import annotations
import errno
import io
import os
import selectors
import socket
import socketserver
import sys
import typing as t
from datetime import datetime as dt
from datetime import timedelta
from datetime import timezone
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from urllib.parse import unquote
from urllib.parse import urlsplit
from ._internal import _log
from ._internal import _wsgi_encoding_dance
from .exceptions import InternalServerError
from .urls import uri_to_iri
class BaseWSGIServer(HTTPServer):
    """A WSGI server that that handles one request at a time.

    Use :func:`make_server` to create a server instance.
    """
    multithread = False
    multiprocess = False
    request_queue_size = LISTEN_QUEUE
    allow_reuse_address = True

    def __init__(self, host: str, port: int, app: WSGIApplication, handler: type[WSGIRequestHandler] | None=None, passthrough_errors: bool=False, ssl_context: _TSSLContextArg | None=None, fd: int | None=None) -> None:
        if handler is None:
            handler = WSGIRequestHandler
        if 'protocol_version' not in vars(handler) and (self.multithread or self.multiprocess):
            handler.protocol_version = 'HTTP/1.1'
        self.host = host
        self.port = port
        self.app = app
        self.passthrough_errors = passthrough_errors
        self.address_family = address_family = select_address_family(host, port)
        server_address = get_sockaddr(host, int(port), address_family)
        if address_family == af_unix and fd is None:
            server_address = t.cast(str, server_address)
            if os.path.exists(server_address):
                os.unlink(server_address)
        super().__init__(server_address, handler, bind_and_activate=False)
        if fd is None:
            try:
                self.server_bind()
                self.server_activate()
            except OSError as e:
                self.server_close()
                print(e.strerror, file=sys.stderr)
                if e.errno == errno.EADDRINUSE:
                    print(f'Port {port} is in use by another program. Either identify and stop that program, or start the server with a different port.', file=sys.stderr)
                    if sys.platform == 'darwin' and port == 5000:
                        print("On macOS, try disabling the 'AirPlay Receiver' service from System Preferences -> General -> AirDrop & Handoff.", file=sys.stderr)
                sys.exit(1)
            except BaseException:
                self.server_close()
                raise
        else:
            self.server_close()
            self.socket = socket.fromfd(fd, address_family, socket.SOCK_STREAM)
            self.server_address = self.socket.getsockname()
        if address_family != af_unix:
            self.port = self.server_address[1]
        if ssl_context is not None:
            if isinstance(ssl_context, tuple):
                ssl_context = load_ssl_context(*ssl_context)
            elif ssl_context == 'adhoc':
                ssl_context = generate_adhoc_ssl_context()
            self.socket = ssl_context.wrap_socket(self.socket, server_side=True)
            self.ssl_context: ssl.SSLContext | None = ssl_context
        else:
            self.ssl_context = None
        import importlib.metadata
        self._server_version = f'Werkzeug/{importlib.metadata.version('werkzeug')}'

    def log(self, type: str, message: str, *args: t.Any) -> None:
        _log(type, message, *args)

    def serve_forever(self, poll_interval: float=0.5) -> None:
        try:
            super().serve_forever(poll_interval=poll_interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.server_close()

    def handle_error(self, request: t.Any, client_address: tuple[str, int] | str) -> None:
        if self.passthrough_errors:
            raise
        return super().handle_error(request, client_address)

    def log_startup(self) -> None:
        """Show information about the address when starting the server."""
        dev_warning = 'WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.'
        dev_warning = _ansi_style(dev_warning, 'bold', 'red')
        messages = [dev_warning]
        if self.address_family == af_unix:
            messages.append(f' * Running on {self.host}')
        else:
            scheme = 'http' if self.ssl_context is None else 'https'
            display_hostname = self.host
            if self.host in {'0.0.0.0', '::'}:
                messages.append(f' * Running on all addresses ({self.host})')
                if self.host == '0.0.0.0':
                    localhost = '127.0.0.1'
                    display_hostname = get_interface_ip(socket.AF_INET)
                else:
                    localhost = '[::1]'
                    display_hostname = get_interface_ip(socket.AF_INET6)
                messages.append(f' * Running on {scheme}://{localhost}:{self.port}')
            if ':' in display_hostname:
                display_hostname = f'[{display_hostname}]'
            messages.append(f' * Running on {scheme}://{display_hostname}:{self.port}')
        _log('info', '\n'.join(messages))