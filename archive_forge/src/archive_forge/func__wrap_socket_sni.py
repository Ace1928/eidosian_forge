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
def _wrap_socket_sni(self, sock, keyfile=None, certfile=None, server_side=False, cert_reqs=None, ca_certs=None, do_handshake_on_connect=False, suppress_ragged_eofs=True, server_hostname=None, ciphers=None, ssl_version=None):
    """Socket wrap with SNI headers.

        stdlib :attr:`ssl.SSLContext.wrap_socket` method augmented with support
        for setting the server_hostname field required for SNI hostname header.

        PARAMETERS:
            sock: socket.socket

                Socket to be wrapped.

            keyfile: str

                Path to the private key

            certfile: str

                Path to the certificate

            server_side: bool

                Identifies whether server-side or client-side
                behavior is desired from this socket. See
                :attr:`~ssl.SSLContext.wrap_socket` for details.

            cert_reqs: ssl.VerifyMode

                When set to other than :attr:`ssl.CERT_NONE`, peers certificate
                is checked. Possible values are :attr:`ssl.CERT_NONE`,
                :attr:`ssl.CERT_OPTIONAL` and :attr:`ssl.CERT_REQUIRED`.

            ca_certs: str

                Path to “certification authority” (CA) certificates
                used to validate other peers’ certificates when ``cert_reqs``
                is other than :attr:`ssl.CERT_NONE`.

            do_handshake_on_connect: bool

                Specifies whether to do the SSL
                handshake automatically. See
                :attr:`~ssl.SSLContext.wrap_socket` for details.

            suppress_ragged_eofs (bool):

                See :attr:`~ssl.SSLContext.wrap_socket` for details.

            server_hostname: str

                Specifies the hostname of the service which
                we are connecting to. See :attr:`~ssl.SSLContext.wrap_socket`
                for details.

            ciphers: str

                Available ciphers for sockets created with this
                context. See :attr:`ssl.SSLContext.set_ciphers`

            ssl_version:

                Protocol of the SSL Context. The value is one of
                ``ssl.PROTOCOL_*`` constants.
        """
    opts = {'sock': sock, 'server_side': server_side, 'do_handshake_on_connect': do_handshake_on_connect, 'suppress_ragged_eofs': suppress_ragged_eofs, 'server_hostname': server_hostname}
    if ssl_version is None:
        ssl_version = ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT
    context = ssl.SSLContext(ssl_version)
    if certfile is not None:
        context.load_cert_chain(certfile, keyfile)
    if ca_certs is not None:
        context.load_verify_locations(ca_certs)
    if ciphers is not None:
        context.set_ciphers(ciphers)
    try:
        context.check_hostname = ssl.HAS_SNI and server_hostname is not None
    except AttributeError:
        pass
    if cert_reqs is not None:
        context.verify_mode = cert_reqs
    if ca_certs is None and context.verify_mode != ssl.CERT_NONE:
        purpose = ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH
        context.load_default_certs(purpose)
    sock = context.wrap_socket(**opts)
    return sock