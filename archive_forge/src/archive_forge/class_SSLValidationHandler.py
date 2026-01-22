from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
class SSLValidationHandler(urllib_request.BaseHandler):
    """
    A custom handler class for SSL validation.

    Based on:
    http://stackoverflow.com/questions/1087227/validate-ssl-certificates-with-python
    http://techknack.net/python-urllib2-handlers/
    """
    CONNECT_COMMAND = 'CONNECT %s:%s HTTP/1.0\r\n'

    def __init__(self, hostname, port, ca_path=None, ciphers=None, validate_certs=True):
        self.hostname = hostname
        self.port = port
        self.ca_path = ca_path
        self.ciphers = ciphers
        self.validate_certs = validate_certs

    def get_ca_certs(self):
        return get_ca_certs(self.ca_path)

    def validate_proxy_response(self, response, valid_codes=None):
        """
        make sure we get back a valid code from the proxy
        """
        valid_codes = [200] if valid_codes is None else valid_codes
        try:
            http_version, resp_code, msg = re.match(b'(HTTP/\\d\\.\\d) (\\d\\d\\d) (.*)', response).groups()
            if int(resp_code) not in valid_codes:
                raise Exception
        except Exception:
            raise ProxyError('Connection to proxy failed')

    def detect_no_proxy(self, url):
        """
        Detect if the 'no_proxy' environment variable is set and honor those locations.
        """
        env_no_proxy = os.environ.get('no_proxy')
        if env_no_proxy:
            env_no_proxy = env_no_proxy.split(',')
            netloc = urlparse(url).netloc
            for host in env_no_proxy:
                if netloc.endswith(host) or netloc.split(':')[0].endswith(host):
                    return False
        return True

    def make_context(self, cafile, cadata, ciphers=None, validate_certs=True):
        cafile = self.ca_path or cafile
        if self.ca_path:
            cadata = None
        else:
            cadata = cadata or None
        return make_context(cafile=cafile, cadata=cadata, ciphers=ciphers, validate_certs=validate_certs)

    def http_request(self, req):
        tmp_ca_cert_path, cadata, paths_checked = self.get_ca_certs()
        use_proxy = self.detect_no_proxy(req.get_full_url())
        https_proxy = os.environ.get('https_proxy')
        context = None
        try:
            context = self.make_context(tmp_ca_cert_path, cadata, ciphers=self.ciphers, validate_certs=self.validate_certs)
        except NotImplementedError:
            pass
        try:
            if use_proxy and https_proxy:
                proxy_parts = generic_urlparse(urlparse(https_proxy))
                port = proxy_parts.get('port') or 443
                proxy_hostname = proxy_parts.get('hostname', None)
                if proxy_hostname is None or proxy_parts.get('scheme') == '':
                    raise ProxyError("Failed to parse https_proxy environment variable. Please make sure you export https proxy as 'https_proxy=<SCHEME>://<IP_ADDRESS>:<PORT>'")
                s = socket.create_connection((proxy_hostname, port))
                if proxy_parts.get('scheme') == 'http':
                    s.sendall(to_bytes(self.CONNECT_COMMAND % (self.hostname, self.port), errors='surrogate_or_strict'))
                    if proxy_parts.get('username'):
                        credentials = '%s:%s' % (proxy_parts.get('username', ''), proxy_parts.get('password', ''))
                        s.sendall(b'Proxy-Authorization: Basic %s\r\n' % base64.b64encode(to_bytes(credentials, errors='surrogate_or_strict')).strip())
                    s.sendall(b'\r\n')
                    connect_result = b''
                    while connect_result.find(b'\r\n\r\n') <= 0:
                        connect_result += s.recv(4096)
                        if len(connect_result) > 131072:
                            raise ProxyError('Proxy sent too verbose headers. Only 128KiB allowed.')
                    self.validate_proxy_response(connect_result)
                    if context:
                        ssl_s = context.wrap_socket(s, server_hostname=self.hostname)
                    elif HAS_URLLIB3_SSL_WRAP_SOCKET:
                        ssl_s = ssl_wrap_socket(s, ca_certs=tmp_ca_cert_path, cert_reqs=ssl.CERT_REQUIRED, ssl_version=PROTOCOL, server_hostname=self.hostname)
                    else:
                        ssl_s = ssl.wrap_socket(s, ca_certs=tmp_ca_cert_path, cert_reqs=ssl.CERT_REQUIRED, ssl_version=PROTOCOL)
                        match_hostname(ssl_s.getpeercert(), self.hostname)
                else:
                    raise ProxyError('Unsupported proxy scheme: %s. Currently ansible only supports HTTP proxies.' % proxy_parts.get('scheme'))
            else:
                s = socket.create_connection((self.hostname, self.port))
                if context:
                    ssl_s = context.wrap_socket(s, server_hostname=self.hostname)
                elif HAS_URLLIB3_SSL_WRAP_SOCKET:
                    ssl_s = ssl_wrap_socket(s, ca_certs=tmp_ca_cert_path, cert_reqs=ssl.CERT_REQUIRED, ssl_version=PROTOCOL, server_hostname=self.hostname)
                else:
                    ssl_s = ssl.wrap_socket(s, ca_certs=tmp_ca_cert_path, cert_reqs=ssl.CERT_REQUIRED, ssl_version=PROTOCOL)
                    match_hostname(ssl_s.getpeercert(), self.hostname)
            s.close()
        except (ssl.SSLError, CertificateError) as e:
            build_ssl_validation_error(self.hostname, self.port, paths_checked, e)
        except socket.error as e:
            raise ConnectionError('Failed to connect to %s at port %s: %s' % (self.hostname, self.port, to_native(e)))
        return req
    https_request = http_request