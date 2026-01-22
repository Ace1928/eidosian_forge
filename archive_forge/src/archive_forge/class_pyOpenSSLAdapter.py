import socket
import sys
import threading
import time
from . import Adapter
from .. import errors, server as cheroot_server
from ..makefile import StreamReader, StreamWriter
class pyOpenSSLAdapter(Adapter):
    """A wrapper for integrating pyOpenSSL with Cheroot."""
    certificate = None
    "The file name of the server's TLS certificate."
    private_key = None
    "The file name of the server's private key file."
    certificate_chain = None
    'Optional. The file name of CA\'s intermediate certificate bundle.\n\n    This is needed for cheaper "chained root" TLS certificates,\n    and should be left as :py:data:`None` if not required.'
    context = None
    '\n    An instance of :py:class:`SSL.Context <pyopenssl:OpenSSL.SSL.Context>`.\n    '
    ciphers = None
    'The ciphers list of TLS.'

    def __init__(self, certificate, private_key, certificate_chain=None, ciphers=None):
        """Initialize OpenSSL Adapter instance."""
        if SSL is None:
            raise ImportError('You must install pyOpenSSL to use HTTPS.')
        super(pyOpenSSLAdapter, self).__init__(certificate, private_key, certificate_chain, ciphers)
        self._environ = None

    def bind(self, sock):
        """Wrap and return the given socket."""
        if self.context is None:
            self.context = self.get_context()
        conn = SSLConnection(self.context, sock)
        self._environ = self.get_environ()
        return conn

    def wrap(self, sock):
        """Wrap and return the given socket, plus WSGI environ entries."""
        return (sock, self._environ.copy())

    def get_context(self):
        """Return an ``SSL.Context`` from self attributes.

        Ref: :py:class:`SSL.Context <pyopenssl:OpenSSL.SSL.Context>`
        """
        c = SSL.Context(SSL.SSLv23_METHOD)
        c.use_privatekey_file(self.private_key)
        if self.certificate_chain:
            c.load_verify_locations(self.certificate_chain)
        c.use_certificate_file(self.certificate)
        return c

    def get_environ(self):
        """Return WSGI environ entries to be merged into each request."""
        ssl_environ = {'wsgi.url_scheme': 'https', 'HTTPS': 'on', 'SSL_VERSION_INTERFACE': '%s %s/%s Python/%s' % (cheroot_server.HTTPServer.version, OpenSSL.version.__title__, OpenSSL.version.__version__, sys.version), 'SSL_VERSION_LIBRARY': SSL.SSLeay_version(SSL.SSLEAY_VERSION).decode()}
        if self.certificate:
            with open(self.certificate, 'rb') as cert_file:
                cert = crypto.load_certificate(crypto.FILETYPE_PEM, cert_file.read())
            ssl_environ.update({'SSL_SERVER_M_VERSION': cert.get_version(), 'SSL_SERVER_M_SERIAL': cert.get_serial_number()})
            for prefix, dn in [('I', cert.get_issuer()), ('S', cert.get_subject())]:
                dnstr = str(dn)[18:-2]
                wsgikey = 'SSL_SERVER_%s_DN' % prefix
                ssl_environ[wsgikey] = dnstr
                while dnstr:
                    pos = dnstr.rfind('=')
                    dnstr, value = (dnstr[:pos], dnstr[pos + 1:])
                    pos = dnstr.rfind('/')
                    dnstr, key = (dnstr[:pos], dnstr[pos + 1:])
                    if key and value:
                        wsgikey = 'SSL_SERVER_%s_DN_%s' % (prefix, key)
                        ssl_environ[wsgikey] = value
        return ssl_environ

    def makefile(self, sock, mode='r', bufsize=-1):
        """Return socket file object."""
        cls = SSLFileobjectStreamReader if 'r' in mode else SSLFileobjectStreamWriter
        if SSL and isinstance(sock, ssl_conn_type):
            wrapped_socket = cls(sock, mode, bufsize)
            wrapped_socket.ssl_timeout = sock.gettimeout()
            return wrapped_socket
        else:
            return cheroot_server.CP_fileobject(sock, mode, bufsize)