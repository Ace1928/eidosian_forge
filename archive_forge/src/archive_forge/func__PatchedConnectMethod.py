from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
import tempfile
import textwrap
import six
import boto
from boto import config
import boto.auth
from boto.exception import NoAuthHandlerFound
from boto.gs.connection import GSConnection
from boto.provider import Provider
from boto.pyami.config import BotoConfigLocations
import gslib
from gslib import context_config
from gslib.exception import CommandException
from gslib.utils import system_util
from gslib.utils.constants import DEFAULT_GCS_JSON_API_VERSION
from gslib.utils.constants import DEFAULT_GSUTIL_STATE_DIR
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import UTF8
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import ONE_MIB
import httplib2
from oauth2client.client import HAS_CRYPTO
def _PatchedConnectMethod(self):
    if hasattr(self, 'timeout'):
        sock = socket.create_connection((self.host, self.port), self.timeout)
    else:
        sock = socket.create_connection((self.host, self.port))
    msg = 'wrapping ssl socket; '
    if self.ca_certs:
        msg += 'CA certificate file=%s' % self.ca_certs
    else:
        msg += 'using system provided SSL certs'
    boto.log.debug(msg)
    if hasattr(ssl, 'SSLContext') and getattr(ssl, 'HAS_SNI', False):
        context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
        context.verify_mode = ssl.CERT_REQUIRED
        if self.ca_certs:
            context.load_verify_locations(self.ca_certs)
        if self.cert_file:
            context.load_cert_chain(self.cert_file, self.key_file)
        self.sock = context.wrap_socket(sock, server_hostname=self.host)
        self.sock.keyfile = self.key_file
        self.sock.certfile = self.cert_file
        self.sock.cert_reqs = context.verify_mode
        self.sock.ssl_version = ssl.PROTOCOL_SSLv23
        self.sock.ca_certs = self.ca_certs
        self.sock.ciphers = None
    else:
        self.sock = ssl.wrap_socket(sock, keyfile=self.key_file, certfile=self.cert_file, cert_reqs=ssl.CERT_REQUIRED, ca_certs=self.ca_certs)
    cert = self.sock.getpeercert()
    hostname = self.host.split(':', 0)[0]
    if not ValidateCertificateHostname(cert, hostname):
        raise InvalidCertificateException(hostname, cert, 'remote hostname "%s" does not match certificate' % hostname)