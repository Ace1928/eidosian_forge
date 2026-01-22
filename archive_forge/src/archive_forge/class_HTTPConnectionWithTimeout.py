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
class HTTPConnectionWithTimeout(http.client.HTTPConnection):
    """HTTPConnection subclass that supports timeouts

    HTTPConnection subclass that supports timeouts

    All timeouts are in seconds. If None is passed for timeout then
    Python's default timeout for sockets will be used. See for example
    the docs of socket.setdefaulttimeout():
    http://docs.python.org/library/socket.html#socket.setdefaulttimeout
    """

    def __init__(self, host, port=None, timeout=None, proxy_info=None):
        http.client.HTTPConnection.__init__(self, host, port=port, timeout=timeout)
        self.proxy_info = proxy_info
        if proxy_info and (not isinstance(proxy_info, ProxyInfo)):
            self.proxy_info = proxy_info('http')

    def connect(self):
        """Connect to the host and port specified in __init__."""
        if self.proxy_info and socks is None:
            raise ProxiesUnavailableError('Proxy support missing but proxy use was requested!')
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
        socket_err = None
        for res in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            try:
                if use_proxy:
                    self.sock = socks.socksocket(af, socktype, proto)
                    self.sock.setproxy(proxy_type, proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass)
                else:
                    self.sock = socket.socket(af, socktype, proto)
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if has_timeout(self.timeout):
                    self.sock.settimeout(self.timeout)
                if self.debuglevel > 0:
                    print('connect: ({0}, {1}) ************'.format(self.host, self.port))
                    if use_proxy:
                        print('proxy: {0} ************'.format(str((proxy_host, proxy_port, proxy_rdns, proxy_user, proxy_pass, proxy_headers))))
                self.sock.connect((self.host, self.port) + sa[2:])
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