import binascii, errno, random, re, socket, subprocess, sys, time, calendar
from datetime import datetime, timezone, timedelta
from io import DEFAULT_BUFFER_SIZE
class IMAP4_SSL(IMAP4):
    """IMAP4 client class over SSL connection

        Instantiate with: IMAP4_SSL([host[, port[, keyfile[, certfile[, ssl_context[, timeout=None]]]]]])

                host - host's name (default: localhost);
                port - port number (default: standard IMAP4 SSL port);
                keyfile - PEM formatted file that contains your private key (default: None);
                certfile - PEM formatted certificate chain file (default: None);
                ssl_context - a SSLContext object that contains your certificate chain
                              and private key (default: None)
                Note: if ssl_context is provided, then parameters keyfile or
                certfile should not be set otherwise ValueError is raised.
                timeout - socket timeout (default: None) If timeout is not given or is None,
                          the global default socket timeout is used

        for more documentation see the docstring of the parent class IMAP4.
        """

    def __init__(self, host='', port=IMAP4_SSL_PORT, keyfile=None, certfile=None, ssl_context=None, timeout=None):
        if ssl_context is not None and keyfile is not None:
            raise ValueError('ssl_context and keyfile arguments are mutually exclusive')
        if ssl_context is not None and certfile is not None:
            raise ValueError('ssl_context and certfile arguments are mutually exclusive')
        if keyfile is not None or certfile is not None:
            import warnings
            warnings.warn('keyfile and certfile are deprecated, use a custom ssl_context instead', DeprecationWarning, 2)
        self.keyfile = keyfile
        self.certfile = certfile
        if ssl_context is None:
            ssl_context = ssl._create_stdlib_context(certfile=certfile, keyfile=keyfile)
        self.ssl_context = ssl_context
        IMAP4.__init__(self, host, port, timeout)

    def _create_socket(self, timeout):
        sock = IMAP4._create_socket(self, timeout)
        return self.ssl_context.wrap_socket(sock, server_hostname=self.host)

    def open(self, host='', port=IMAP4_SSL_PORT, timeout=None):
        """Setup connection to remote server on "host:port".
                (default: localhost:standard IMAP4 SSL port).
            This connection will be used by the routines:
                read, readline, send, shutdown.
            """
        IMAP4.open(self, host, port, timeout)