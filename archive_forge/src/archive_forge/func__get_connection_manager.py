import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
def _get_connection_manager(self):
    pool_manager_init_args = {'timeout': self.get_timeout()}
    if self._ca_certs:
        pool_manager_init_args['cert_reqs'] = 'CERT_REQUIRED'
        pool_manager_init_args['ca_certs'] = self._ca_certs
    if self._proxy_url:
        if self._proxy_url.lower().startswith('sock'):
            from urllib3.contrib.socks import SOCKSProxyManager
            return SOCKSProxyManager(self._proxy_url, **pool_manager_init_args)
        if self._identify_http_proxy_auth():
            self._proxy_url, self._basic_proxy_auth = self._separate_http_proxy_auth()
            pool_manager_init_args['proxy_headers'] = urllib3.make_headers(proxy_basic_auth=self._basic_proxy_auth)
        return urllib3.ProxyManager(self._proxy_url, **pool_manager_init_args)
    return urllib3.PoolManager(**pool_manager_init_args)