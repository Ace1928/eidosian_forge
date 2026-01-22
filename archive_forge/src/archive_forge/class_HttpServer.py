import errno
import http.client as http_client
import http.server as http_server
import os
import posixpath
import random
import re
import socket
import sys
from urllib.parse import urlparse
from .. import osutils, urlutils
from . import test_server
class HttpServer(test_server.TestingTCPServerInAThread):
    """A test server for http transports.

    Subclasses can provide a specific request handler.
    """
    http_server_class = {'HTTP/1.0': TestingHTTPServer, 'HTTP/1.1': TestingThreadingHTTPServer}
    proxy_requests = False
    _url_protocol = 'http'

    def __init__(self, request_handler=TestingHTTPRequestHandler, protocol_version=None):
        """Constructor.

        :param request_handler: a class that will be instantiated to handle an
            http connection (one or several requests).

        :param protocol_version: if specified, will override the protocol
            version of the request handler.
        """
        if protocol_version is None:
            proto_vers = request_handler.protocol_version
        else:
            proto_vers = protocol_version
        serv_cls = self.http_server_class.get(proto_vers, None)
        if serv_cls is None:
            raise http_client.UnknownProtocol(proto_vers)
        self.host = 'localhost'
        self.port = 0
        super().__init__((self.host, self.port), serv_cls, request_handler)
        self.protocol_version = proto_vers
        self.GET_request_nb = 0
        self._http_base_url = None
        self.logs = []

    def create_server(self):
        return self.server_class((self.host, self.port), self.request_handler_class, self)

    def _get_remote_url(self, path):
        path_parts = path.split(os.path.sep)
        if os.path.isabs(path):
            if path_parts[:len(self._local_path_parts)] != self._local_path_parts:
                raise BadWebserverPath(path, self.test_dir)
            remote_path = '/'.join(path_parts[len(self._local_path_parts):])
        else:
            remote_path = '/'.join(path_parts)
        return self._http_base_url + remote_path

    def log(self, format, *args):
        """Capture Server log output."""
        self.logs.append(format % args)

    def start_server(self, backing_transport_server=None):
        """See breezy.transport.Server.start_server.

        :param backing_transport_server: The transport that requests over this
            protocol should be forwarded to. Note that this is currently not
            supported for HTTP.
        """
        if not (backing_transport_server is None or isinstance(backing_transport_server, test_server.LocalURLServer)):
            raise AssertionError('HTTPServer currently assumes local transport, got %s' % backing_transport_server)
        self._home_dir = osutils.getcwd()
        self._local_path_parts = self._home_dir.split(os.path.sep)
        self.logs = []
        super().start_server()
        self._http_base_url = '{}://{}:{}/'.format(self._url_protocol, self.host, self.port)

    def get_url(self):
        """See breezy.transport.Server.get_url."""
        return self._get_remote_url(self._home_dir)

    def get_bogus_url(self):
        """See breezy.transport.Server.get_bogus_url."""
        return self._url_protocol + '://127.0.0.1:1/'