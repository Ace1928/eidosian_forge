import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def _open_generic_http(self, connection_factory, url, data):
    """Make an HTTP connection using connection_class.

        This is an internal method that should be called from
        open_http() or open_https().

        Arguments:
        - connection_factory should take a host name and return an
          HTTPConnection instance.
        - url is the url to retrieval or a host, relative-path pair.
        - data is payload for a POST request or None.
        """
    user_passwd = None
    proxy_passwd = None
    if isinstance(url, str):
        host, selector = _splithost(url)
        if host:
            user_passwd, host = _splituser(host)
            host = unquote(host)
        realhost = host
    else:
        host, selector = url
        proxy_passwd, host = _splituser(host)
        urltype, rest = _splittype(selector)
        url = rest
        user_passwd = None
        if urltype.lower() != 'http':
            realhost = None
        else:
            realhost, rest = _splithost(rest)
            if realhost:
                user_passwd, realhost = _splituser(realhost)
            if user_passwd:
                selector = '%s://%s%s' % (urltype, realhost, rest)
            if proxy_bypass(realhost):
                host = realhost
    if not host:
        raise OSError('http error', 'no host given')
    if proxy_passwd:
        proxy_passwd = unquote(proxy_passwd)
        proxy_auth = base64.b64encode(proxy_passwd.encode()).decode('ascii')
    else:
        proxy_auth = None
    if user_passwd:
        user_passwd = unquote(user_passwd)
        auth = base64.b64encode(user_passwd.encode()).decode('ascii')
    else:
        auth = None
    http_conn = connection_factory(host)
    headers = {}
    if proxy_auth:
        headers['Proxy-Authorization'] = 'Basic %s' % proxy_auth
    if auth:
        headers['Authorization'] = 'Basic %s' % auth
    if realhost:
        headers['Host'] = realhost
    headers['Connection'] = 'close'
    for header, value in self.addheaders:
        headers[header] = value
    if data is not None:
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        http_conn.request('POST', selector, data, headers)
    else:
        http_conn.request('GET', selector, headers=headers)
    try:
        response = http_conn.getresponse()
    except http.client.BadStatusLine:
        raise URLError('http protocol error: bad status line')
    if 200 <= response.status < 300:
        return addinfourl(response, response.msg, 'http:' + url, response.status)
    else:
        return self.http_error(url, response.fp, response.status, response.reason, response.msg, data)