import logging
import sys
import io
import cheroot.server
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
from ._cpcompat import tonative
class CPHTTPServer(cheroot.server.HTTPServer):
    """Wrapper for cheroot.server.HTTPServer.

    cheroot has been designed to not reference CherryPy in any way,
    so that it can be used in other frameworks and applications.
    Therefore, we wrap it here, so we can apply some attributes
    from config -> cherrypy.server -> HTTPServer.
    """

    def __init__(self, server_adapter=cherrypy.server):
        """Initialize CPHTTPServer."""
        self.server_adapter = server_adapter
        server_name = self.server_adapter.socket_host or self.server_adapter.socket_file or None
        cheroot.server.HTTPServer.__init__(self, server_adapter.bind_addr, NativeGateway, minthreads=server_adapter.thread_pool, maxthreads=server_adapter.thread_pool_max, server_name=server_name)
        self.max_request_header_size = self.server_adapter.max_request_header_size or 0
        self.max_request_body_size = self.server_adapter.max_request_body_size or 0
        self.request_queue_size = self.server_adapter.socket_queue_size
        self.timeout = self.server_adapter.socket_timeout
        self.shutdown_timeout = self.server_adapter.shutdown_timeout
        self.protocol = self.server_adapter.protocol_version
        self.nodelay = self.server_adapter.nodelay
        ssl_module = self.server_adapter.ssl_module or 'pyopenssl'
        if self.server_adapter.ssl_context:
            adapter_class = cheroot.server.get_ssl_adapter_class(ssl_module)
            self.ssl_adapter = adapter_class(self.server_adapter.ssl_certificate, self.server_adapter.ssl_private_key, self.server_adapter.ssl_certificate_chain, self.server_adapter.ssl_ciphers)
            self.ssl_adapter.context = self.server_adapter.ssl_context
        elif self.server_adapter.ssl_certificate:
            adapter_class = cheroot.server.get_ssl_adapter_class(ssl_module)
            self.ssl_adapter = adapter_class(self.server_adapter.ssl_certificate, self.server_adapter.ssl_private_key, self.server_adapter.ssl_certificate_chain, self.server_adapter.ssl_ciphers)