import base64
import calendar
import copy
import email
import email.feedparser
from email import header
import email.message
import email.utils
import errno
from gettext import gettext as _
import gzip
from hashlib import md5 as _md5
from hashlib import sha1 as _sha
import hmac
import http.client
import io
import os
import random
import re
import socket
import ssl
import sys
import time
import urllib.parse
import zlib
from . import auth
from .error import *
from .iri2uri import iri2uri
from httplib2 import certs
class HTTPSConnectionWithTimeout(http.client.HTTPSConnection):
    """This class allows communication via SSL.

    All timeouts are in seconds. If None is passed for timeout then
    Python's default timeout for sockets will be used. See for example
    the docs of socket.setdefaulttimeout():
    http://docs.python.org/library/socket.html#socket.setdefaulttimeout
    """

    def __init__(self, host, port=None, key_file=None, cert_file=None, timeout=None, proxy_info=None, ca_certs=None, disable_ssl_certificate_validation=False, tls_maximum_version=None, tls_minimum_version=None, key_password=None):
        self.disable_ssl_certificate_validation = disable_ssl_certificate_validation
        self.ca_certs = ca_certs if ca_certs else CA_CERTS
        self.proxy_info = proxy_info
        if proxy_info and (not isinstance(proxy_info, ProxyInfo)):
            self.proxy_info = proxy_info('https')
        context = _build_ssl_context(self.disable_ssl_certificate_validation, self.ca_certs, cert_file, key_file, maximum_version=tls_maximum_version, minimum_version=tls_minimum_version, key_password=key_password)
        super(HTTPSConnectionWithTimeout, self).__init__(host, port=port, timeout=timeout, context=context)
        self.key_file = key_file
        self.cert_file = cert_file
        self.key_password = key_password

    def connect(self):
        """Connect to a host on a given (SSL) port."""
        if self.proxy_info and self.proxy_info.isgood() and self.proxy_info.applies_to(self.host):
            use_proxy = True
            proxy_type, proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers = self.proxy_info.astuple()
            host = proxy_host
            port = proxy_port
        else:
            use_proxy = False
            host = self.host
            port = self.port
            proxy_type = None
            proxy_headers = None
        socket_err = None
        address_info = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        for family, socktype, proto, canonname, sockaddr in address_info:
            try:
                if use_proxy:
                    sock = socks.socksocket(family, socktype, proto)
                    sock.setproxy(proxy_type, proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass)
                else:
                    sock = socket.socket(family, socktype, proto)
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if has_timeout(self.timeout):
                    sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                if not hasattr(self._context, 'check_hostname') and (not self.disable_ssl_certificate_validation):
                    try:
                        ssl.match_hostname(self.sock.getpeercert(), self.host)
                    except Exception:
                        self.sock.shutdown(socket.SHUT_RDWR)
                        self.sock.close()
                        raise
                if self.debuglevel > 0:
                    print('connect: ({0}, {1})'.format(self.host, self.port))
                    if use_proxy:
                        print('proxy: {0}'.format(str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers))))
            except (ssl.SSLError, ssl.CertificateError) as e:
                if sock:
                    sock.close()
                if self.sock:
                    self.sock.close()
                self.sock = None
                raise
            except (socket.timeout, socket.gaierror):
                raise
            except socket.error as e:
                socket_err = e
                if self.debuglevel > 0:
                    print('connect fail: ({0}, {1})'.format(self.host, self.port))
                    if use_proxy:
                        print('proxy: {0}'.format(str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers))))
                if self.sock:
                    self.sock.close()
                self.sock = None
                continue
            break
        if not self.sock:
            raise socket_err