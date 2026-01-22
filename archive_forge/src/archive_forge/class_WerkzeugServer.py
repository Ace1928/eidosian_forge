from abc import ABCMeta
from abc import abstractmethod
import argparse
import atexit
from collections import defaultdict
import errno
import logging
import mimetypes
import os
import shlex
import signal
import socket
import sys
import threading
import time
import urllib.parse
from absl import flags as absl_flags
from absl.flags import argparse_flags
from werkzeug import serving
from tensorboard import manager
from tensorboard import version
from tensorboard.backend import application
from tensorboard.backend.event_processing import data_ingester as local_ingester
from tensorboard.backend.event_processing import event_file_inspector as efi
from tensorboard.data import server_ingester
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
class WerkzeugServer(serving.ThreadedWSGIServer, TensorBoardServer):
    """Implementation of TensorBoardServer using the Werkzeug dev server."""
    daemon_threads = True

    def __init__(self, wsgi_app, flags):
        self._flags = flags
        host = flags.host
        port = flags.port
        self._auto_wildcard = flags.bind_all
        if self._auto_wildcard:
            host = self._get_wildcard_address(port)
        elif host is None:
            host = 'localhost'
        self._host = host
        self._url = None
        self._fix_werkzeug_logging()
        try:
            super().__init__(host, port, wsgi_app, _WSGIRequestHandler)
        except socket.error as e:
            if hasattr(errno, 'EACCES') and e.errno == errno.EACCES:
                raise TensorBoardServerException('TensorBoard must be run as superuser to bind to port %d' % port)
            elif hasattr(errno, 'EADDRINUSE') and e.errno == errno.EADDRINUSE:
                if port == 0:
                    raise TensorBoardServerException('TensorBoard unable to find any open port')
                else:
                    raise TensorBoardPortInUseError('TensorBoard could not bind to port %d, it was already in use' % port)
            elif hasattr(errno, 'EADDRNOTAVAIL') and e.errno == errno.EADDRNOTAVAIL:
                raise TensorBoardServerException('TensorBoard could not bind to unavailable address %s' % host)
            elif hasattr(errno, 'EAFNOSUPPORT') and e.errno == errno.EAFNOSUPPORT:
                raise TensorBoardServerException('Tensorboard could not bind to unsupported address family %s' % host)
            raise

    def _get_wildcard_address(self, port):
        """Returns a wildcard address for the port in question.

        This will attempt to follow the best practice of calling
        getaddrinfo() with a null host and AI_PASSIVE to request a
        server-side socket wildcard address. If that succeeds, this
        returns the first IPv6 address found, or if none, then returns
        the first IPv4 address. If that fails, then this returns the
        hardcoded address "::" if socket.has_ipv6 is True, else
        "0.0.0.0".
        """
        fallback_address = '::' if socket.has_ipv6 else '0.0.0.0'
        if hasattr(socket, 'AI_PASSIVE'):
            try:
                addrinfos = socket.getaddrinfo(None, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP, socket.AI_PASSIVE)
            except socket.gaierror as e:
                logger.warning('Failed to auto-detect wildcard address, assuming %s: %s', fallback_address, str(e))
                return fallback_address
            addrs_by_family = defaultdict(list)
            for family, _, _, _, sockaddr in addrinfos:
                addrs_by_family[family].append(sockaddr[0])
            if hasattr(socket, 'AF_INET6') and addrs_by_family[socket.AF_INET6]:
                return addrs_by_family[socket.AF_INET6][0]
            if hasattr(socket, 'AF_INET') and addrs_by_family[socket.AF_INET]:
                return addrs_by_family[socket.AF_INET][0]
        logger.warning('Failed to auto-detect wildcard address, assuming %s', fallback_address)
        return fallback_address

    def server_bind(self):
        """Override to set custom options on the socket."""
        if self._flags.reuse_port:
            try:
                socket.SO_REUSEPORT
            except AttributeError:
                raise TensorBoardServerException('TensorBoard --reuse_port option is not supported on this platform')
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        socket_is_v6 = hasattr(socket, 'AF_INET6') and self.socket.family == socket.AF_INET6
        has_v6only_option = hasattr(socket, 'IPPROTO_IPV6') and hasattr(socket, 'IPV6_V6ONLY')
        if self._auto_wildcard and socket_is_v6 and has_v6only_option:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except socket.error as e:
                if hasattr(errno, 'EAFNOSUPPORT') and e.errno != errno.EAFNOSUPPORT:
                    logger.warning('Failed to dual-bind to IPv4 wildcard: %s', str(e))
        super().server_bind()

    def handle_error(self, request, client_address):
        """Override to get rid of noisy EPIPE errors."""
        del request
        exc_info = sys.exc_info()
        e = exc_info[1]
        if isinstance(e, IOError) and e.errno == errno.EPIPE:
            logger.warning('EPIPE caused by %s in HTTP serving' % str(client_address))
        else:
            logger.error('HTTP serving error', exc_info=exc_info)

    def get_url(self):
        if not self._url:
            if self._auto_wildcard:
                display_host = socket.getfqdn()
                try:
                    socket.create_connection((display_host, self.server_port), timeout=1)
                except socket.error as e:
                    display_host = 'localhost'
            else:
                host = self._host
                display_host = '[%s]' % host if ':' in host and (not host.startswith('[')) else host
            self._url = 'http://%s:%d%s/' % (display_host, self.server_port, self._flags.path_prefix.rstrip('/'))
        return self._url

    def print_serving_message(self):
        if self._flags.host is None and (not self._flags.bind_all):
            sys.stderr.write('Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all\n')
            sys.stderr.flush()
        super().print_serving_message()

    def _fix_werkzeug_logging(self):
        """Fix werkzeug logging setup so it inherits TensorBoard's log level.

        This addresses a change in werkzeug 0.15.0+ [1] that causes it set its own
        log level to INFO regardless of the root logger configuration. We instead
        want werkzeug to inherit TensorBoard's root logger log level (set via absl
        to WARNING by default).

        [1]: https://github.com/pallets/werkzeug/commit/4cf77d25858ff46ac7e9d64ade054bf05b41ce12
        """
        self.log('debug', 'Fixing werkzeug logger to inherit TensorBoard log level')
        logging.getLogger('werkzeug').setLevel(logging.NOTSET)