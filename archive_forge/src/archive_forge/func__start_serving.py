import io
import os
import socket
import warnings
import signal
import threading
import collections
from . import base_events
from . import constants
from . import futures
from . import exceptions
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _start_serving(self, protocol_factory, sock, sslcontext=None, server=None, backlog=100, ssl_handshake_timeout=None, ssl_shutdown_timeout=None):

    def loop(f=None):
        try:
            if f is not None:
                conn, addr = f.result()
                if self._debug:
                    logger.debug('%r got a new connection from %r: %r', server, addr, conn)
                protocol = protocol_factory()
                if sslcontext is not None:
                    self._make_ssl_transport(conn, protocol, sslcontext, server_side=True, extra={'peername': addr}, server=server, ssl_handshake_timeout=ssl_handshake_timeout, ssl_shutdown_timeout=ssl_shutdown_timeout)
                else:
                    self._make_socket_transport(conn, protocol, extra={'peername': addr}, server=server)
            if self.is_closed():
                return
            f = self._proactor.accept(sock)
        except OSError as exc:
            if sock.fileno() != -1:
                self.call_exception_handler({'message': 'Accept failed on a socket', 'exception': exc, 'socket': trsock.TransportSocket(sock)})
                sock.close()
            elif self._debug:
                logger.debug('Accept failed on socket %r', sock, exc_info=True)
        except exceptions.CancelledError:
            sock.close()
        else:
            self._accept_futures[sock.fileno()] = f
            f.add_done_callback(loop)
    self.call_soon(loop)