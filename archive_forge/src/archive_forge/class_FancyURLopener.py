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
class FancyURLopener(URLopener):
    """Derived class with handlers for errors we can handle (perhaps)."""

    def __init__(self, *args, **kwargs):
        URLopener.__init__(self, *args, **kwargs)
        self.auth_cache = {}
        self.tries = 0
        self.maxtries = 10

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        """Default error handling -- don't raise an exception."""
        return addinfourl(fp, headers, 'http:' + url, errcode)

    def http_error_302(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 302 -- relocated (temporarily)."""
        self.tries += 1
        try:
            if self.maxtries and self.tries >= self.maxtries:
                if hasattr(self, 'http_error_500'):
                    meth = self.http_error_500
                else:
                    meth = self.http_error_default
                return meth(url, fp, 500, 'Internal Server Error: Redirect Recursion', headers)
            result = self.redirect_internal(url, fp, errcode, errmsg, headers, data)
            return result
        finally:
            self.tries = 0

    def redirect_internal(self, url, fp, errcode, errmsg, headers, data):
        if 'location' in headers:
            newurl = headers['location']
        elif 'uri' in headers:
            newurl = headers['uri']
        else:
            return
        fp.close()
        newurl = urljoin(self.type + ':' + url, newurl)
        urlparts = urlparse(newurl)
        if urlparts.scheme not in ('http', 'https', 'ftp', ''):
            raise HTTPError(newurl, errcode, errmsg + " Redirection to url '%s' is not allowed." % newurl, headers, fp)
        return self.open(newurl)

    def http_error_301(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 301 -- also relocated (permanently)."""
        return self.http_error_302(url, fp, errcode, errmsg, headers, data)

    def http_error_303(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 303 -- also relocated (essentially identical to 302)."""
        return self.http_error_302(url, fp, errcode, errmsg, headers, data)

    def http_error_307(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 307 -- relocated, but turn POST into error."""
        if data is None:
            return self.http_error_302(url, fp, errcode, errmsg, headers, data)
        else:
            return self.http_error_default(url, fp, errcode, errmsg, headers)

    def http_error_308(self, url, fp, errcode, errmsg, headers, data=None):
        """Error 308 -- relocated, but turn POST into error."""
        if data is None:
            return self.http_error_301(url, fp, errcode, errmsg, headers, data)
        else:
            return self.http_error_default(url, fp, errcode, errmsg, headers)

    def http_error_401(self, url, fp, errcode, errmsg, headers, data=None, retry=False):
        """Error 401 -- authentication required.
        This function supports Basic authentication only."""
        if 'www-authenticate' not in headers:
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        stuff = headers['www-authenticate']
        match = re.match('[ \t]*([^ \t]+)[ \t]+realm="([^"]*)"', stuff)
        if not match:
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        scheme, realm = match.groups()
        if scheme.lower() != 'basic':
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        if not retry:
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        name = 'retry_' + self.type + '_basic_auth'
        if data is None:
            return getattr(self, name)(url, realm)
        else:
            return getattr(self, name)(url, realm, data)

    def http_error_407(self, url, fp, errcode, errmsg, headers, data=None, retry=False):
        """Error 407 -- proxy authentication required.
        This function supports Basic authentication only."""
        if 'proxy-authenticate' not in headers:
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        stuff = headers['proxy-authenticate']
        match = re.match('[ \t]*([^ \t]+)[ \t]+realm="([^"]*)"', stuff)
        if not match:
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        scheme, realm = match.groups()
        if scheme.lower() != 'basic':
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        if not retry:
            URLopener.http_error_default(self, url, fp, errcode, errmsg, headers)
        name = 'retry_proxy_' + self.type + '_basic_auth'
        if data is None:
            return getattr(self, name)(url, realm)
        else:
            return getattr(self, name)(url, realm, data)

    def retry_proxy_http_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        newurl = 'http://' + host + selector
        proxy = self.proxies['http']
        urltype, proxyhost = _splittype(proxy)
        proxyhost, proxyselector = _splithost(proxyhost)
        i = proxyhost.find('@') + 1
        proxyhost = proxyhost[i:]
        user, passwd = self.get_user_passwd(proxyhost, realm, i)
        if not (user or passwd):
            return None
        proxyhost = '%s:%s@%s' % (quote(user, safe=''), quote(passwd, safe=''), proxyhost)
        self.proxies['http'] = 'http://' + proxyhost + proxyselector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def retry_proxy_https_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        newurl = 'https://' + host + selector
        proxy = self.proxies['https']
        urltype, proxyhost = _splittype(proxy)
        proxyhost, proxyselector = _splithost(proxyhost)
        i = proxyhost.find('@') + 1
        proxyhost = proxyhost[i:]
        user, passwd = self.get_user_passwd(proxyhost, realm, i)
        if not (user or passwd):
            return None
        proxyhost = '%s:%s@%s' % (quote(user, safe=''), quote(passwd, safe=''), proxyhost)
        self.proxies['https'] = 'https://' + proxyhost + proxyselector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def retry_http_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        i = host.find('@') + 1
        host = host[i:]
        user, passwd = self.get_user_passwd(host, realm, i)
        if not (user or passwd):
            return None
        host = '%s:%s@%s' % (quote(user, safe=''), quote(passwd, safe=''), host)
        newurl = 'http://' + host + selector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def retry_https_basic_auth(self, url, realm, data=None):
        host, selector = _splithost(url)
        i = host.find('@') + 1
        host = host[i:]
        user, passwd = self.get_user_passwd(host, realm, i)
        if not (user or passwd):
            return None
        host = '%s:%s@%s' % (quote(user, safe=''), quote(passwd, safe=''), host)
        newurl = 'https://' + host + selector
        if data is None:
            return self.open(newurl)
        else:
            return self.open(newurl, data)

    def get_user_passwd(self, host, realm, clear_cache=0):
        key = realm + '@' + host.lower()
        if key in self.auth_cache:
            if clear_cache:
                del self.auth_cache[key]
            else:
                return self.auth_cache[key]
        user, passwd = self.prompt_user_passwd(host, realm)
        if user or passwd:
            self.auth_cache[key] = (user, passwd)
        return (user, passwd)

    def prompt_user_passwd(self, host, realm):
        """Override this in a GUI environment!"""
        import getpass
        try:
            user = input('Enter username for %s at %s: ' % (realm, host))
            passwd = getpass.getpass('Enter password for %s in %s at %s: ' % (user, realm, host))
            return (user, passwd)
        except KeyboardInterrupt:
            print()
            return (None, None)