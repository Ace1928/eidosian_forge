import socket
import sys
import threading
from contextlib import suppress
from . import Adapter
from .. import errors
from .._compat import IS_ABOVE_OPENSSL10
from ..makefile import StreamReader, StreamWriter
from ..server import HTTPServer
def _make_env_dn_dict(self, env_prefix, cert_value):
    """Return a dict of WSGI environment variables for a certificate DN.

        E.g. SSL_CLIENT_S_DN_CN, SSL_CLIENT_S_DN_C, etc.
        See SSL_CLIENT_S_DN_x509 at
        https://httpd.apache.org/docs/2.4/mod/mod_ssl.html#envvars.
        """
    if not cert_value:
        return {}
    dn = []
    dn_attrs = {}
    for rdn in cert_value:
        for attr_name, val in rdn:
            attr_code = self.CERT_KEY_TO_LDAP_CODE.get(attr_name)
            dn.append('%s=%s' % (attr_code or attr_name, val))
            if not attr_code:
                continue
            dn_attrs.setdefault(attr_code, [])
            dn_attrs[attr_code].append(val)
    env = {env_prefix: ','.join(dn)}
    for attr_code, values in dn_attrs.items():
        env['%s_%s' % (env_prefix, attr_code)] = ','.join(values)
        if len(values) == 1:
            continue
        for i, val in enumerate(values):
            env['%s_%s_%i' % (env_prefix, attr_code, i)] = val
    return env