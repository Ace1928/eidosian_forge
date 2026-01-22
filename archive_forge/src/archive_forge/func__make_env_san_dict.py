import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _make_env_san_dict(self, env_prefix, cert_value):
    """Return a dict of WSGI environment variables for a certificate DN.

        E.g. SSL_CLIENT_SAN_Email_0, SSL_CLIENT_SAN_DNS_0, etc.
        See SSL_CLIENT_SAN_* at
        https://httpd.apache.org/docs/2.4/mod/mod_ssl.html#envvars.
        """
    if not cert_value:
        return {}
    env = {}
    dns_count = 0
    email_count = 0
    for attr_name, val in cert_value:
        if attr_name == 'DNS':
            env['%s_DNS_%i' % (env_prefix, dns_count)] = val
            dns_count += 1
        elif attr_name == 'Email':
            env['%s_Email_%i' % (env_prefix, email_count)] = val
            email_count += 1
    return env