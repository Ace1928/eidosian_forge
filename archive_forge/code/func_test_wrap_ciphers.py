import os
import ssl
from unittest import mock
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
def test_wrap_ciphers(self):
    self.conf.set_default('ca_file', self.ca_file_name, group=sslutils.config_section)
    ciphers = 'ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:ECDH+HIGH:DH+HIGH:ECDH+3DES:DH+3DES:RSA+AESGCM:RSA+AES:RSA+HIGH:RSA+3DES:!aNULL:!eNULL:!MD5:!DSS:!RC4'
    self.conf.set_default('ciphers', ciphers, group=sslutils.config_section)
    ssl_kwargs = {'ca_certs': self.conf.ssl.ca_file, 'cert_reqs': ssl.CERT_REQUIRED, 'ciphers': ciphers}
    self._test_wrap(**ssl_kwargs)