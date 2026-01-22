import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _make_env_cert_dict(self, env_prefix, parsed_cert):
    """Return a dict of WSGI environment variables for a certificate.

        E.g. SSL_CLIENT_M_VERSION, SSL_CLIENT_M_SERIAL, etc.
        See https://httpd.apache.org/docs/2.4/mod/mod_ssl.html#envvars.
        """
    if not parsed_cert:
        return {}
    env = {}
    for cert_key, env_var in self.CERT_KEY_TO_ENV.items():
        key = '%s_%s' % (env_prefix, env_var)
        value = parsed_cert.get(cert_key)
        if env_var == 'SAN':
            env.update(self._make_env_san_dict(key, value))
        elif env_var.endswith('_DN'):
            env.update(self._make_env_dn_dict(key, value))
        else:
            env[key] = str(value)
    if 'notBefore' in parsed_cert:
        remain = ssl.cert_time_to_seconds(parsed_cert['notAfter'])
        remain -= ssl.cert_time_to_seconds(parsed_cert['notBefore'])
        remain /= 60 * 60 * 24
        env['%s_V_REMAIN' % (env_prefix,)] = str(int(remain))
    return env