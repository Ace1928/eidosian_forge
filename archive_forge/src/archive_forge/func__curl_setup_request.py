import collections
import functools
import logging
import pycurl
import threading
import time
from io import BytesIO
from tornado import httputil
from tornado import ioloop
from tornado.escape import utf8, native_str
from tornado.httpclient import (
from tornado.log import app_log
from typing import Dict, Any, Callable, Union, Optional
import typing
def _curl_setup_request(self, curl: pycurl.Curl, request: HTTPRequest, buffer: BytesIO, headers: httputil.HTTPHeaders) -> None:
    curl.setopt(pycurl.URL, native_str(request.url))
    if 'Expect' not in request.headers:
        request.headers['Expect'] = ''
    if 'Pragma' not in request.headers:
        request.headers['Pragma'] = ''
    curl.setopt(pycurl.HTTPHEADER, [b'%s: %s' % (native_str(k).encode('ASCII'), native_str(v).encode('ISO8859-1')) for k, v in request.headers.get_all()])
    curl.setopt(pycurl.HEADERFUNCTION, functools.partial(self._curl_header_callback, headers, request.header_callback))
    if request.streaming_callback:

        def write_function(b: Union[bytes, bytearray]) -> int:
            assert request.streaming_callback is not None
            self.io_loop.add_callback(request.streaming_callback, b)
            return len(b)
    else:
        write_function = buffer.write
    curl.setopt(pycurl.WRITEFUNCTION, write_function)
    curl.setopt(pycurl.FOLLOWLOCATION, request.follow_redirects)
    curl.setopt(pycurl.MAXREDIRS, request.max_redirects)
    assert request.connect_timeout is not None
    curl.setopt(pycurl.CONNECTTIMEOUT_MS, int(1000 * request.connect_timeout))
    assert request.request_timeout is not None
    curl.setopt(pycurl.TIMEOUT_MS, int(1000 * request.request_timeout))
    if request.user_agent:
        curl.setopt(pycurl.USERAGENT, native_str(request.user_agent))
    else:
        curl.setopt(pycurl.USERAGENT, 'Mozilla/5.0 (compatible; pycurl)')
    if request.network_interface:
        curl.setopt(pycurl.INTERFACE, request.network_interface)
    if request.decompress_response:
        curl.setopt(pycurl.ENCODING, 'gzip,deflate')
    else:
        curl.setopt(pycurl.ENCODING, None)
    if request.proxy_host and request.proxy_port:
        curl.setopt(pycurl.PROXY, request.proxy_host)
        curl.setopt(pycurl.PROXYPORT, request.proxy_port)
        if request.proxy_username:
            assert request.proxy_password is not None
            credentials = httputil.encode_username_password(request.proxy_username, request.proxy_password)
            curl.setopt(pycurl.PROXYUSERPWD, credentials)
        if request.proxy_auth_mode is None or request.proxy_auth_mode == 'basic':
            curl.setopt(pycurl.PROXYAUTH, pycurl.HTTPAUTH_BASIC)
        elif request.proxy_auth_mode == 'digest':
            curl.setopt(pycurl.PROXYAUTH, pycurl.HTTPAUTH_DIGEST)
        else:
            raise ValueError('Unsupported proxy_auth_mode %s' % request.proxy_auth_mode)
    else:
        try:
            curl.unsetopt(pycurl.PROXY)
        except TypeError:
            curl.setopt(pycurl.PROXY, '')
        curl.unsetopt(pycurl.PROXYUSERPWD)
    if request.validate_cert:
        curl.setopt(pycurl.SSL_VERIFYPEER, 1)
        curl.setopt(pycurl.SSL_VERIFYHOST, 2)
    else:
        curl.setopt(pycurl.SSL_VERIFYPEER, 0)
        curl.setopt(pycurl.SSL_VERIFYHOST, 0)
    if request.ca_certs is not None:
        curl.setopt(pycurl.CAINFO, request.ca_certs)
    else:
        pass
    if request.allow_ipv6 is False:
        curl.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V4)
    else:
        curl.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_WHATEVER)
    curl_options = {'GET': pycurl.HTTPGET, 'POST': pycurl.POST, 'PUT': pycurl.UPLOAD, 'HEAD': pycurl.NOBODY}
    custom_methods = set(['DELETE', 'OPTIONS', 'PATCH'])
    for o in curl_options.values():
        curl.setopt(o, False)
    if request.method in curl_options:
        curl.unsetopt(pycurl.CUSTOMREQUEST)
        curl.setopt(curl_options[request.method], True)
    elif request.allow_nonstandard_methods or request.method in custom_methods:
        curl.setopt(pycurl.CUSTOMREQUEST, request.method)
    else:
        raise KeyError('unknown method ' + request.method)
    body_expected = request.method in ('POST', 'PATCH', 'PUT')
    body_present = request.body is not None
    if not request.allow_nonstandard_methods:
        if body_expected and (not body_present) or (body_present and (not body_expected)):
            raise ValueError('Body must %sbe None for method %s (unless allow_nonstandard_methods is true)' % ('not ' if body_expected else '', request.method))
    if body_expected or body_present:
        if request.method == 'GET':
            raise ValueError('Body must be None for GET request')
        request_buffer = BytesIO(utf8(request.body or ''))

        def ioctl(cmd: int) -> None:
            if cmd == curl.IOCMD_RESTARTREAD:
                request_buffer.seek(0)
        curl.setopt(pycurl.READFUNCTION, request_buffer.read)
        curl.setopt(pycurl.IOCTLFUNCTION, ioctl)
        if request.method == 'POST':
            curl.setopt(pycurl.POSTFIELDSIZE, len(request.body or ''))
        else:
            curl.setopt(pycurl.UPLOAD, True)
            curl.setopt(pycurl.INFILESIZE, len(request.body or ''))
    if request.auth_username is not None:
        assert request.auth_password is not None
        if request.auth_mode is None or request.auth_mode == 'basic':
            curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_BASIC)
        elif request.auth_mode == 'digest':
            curl.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_DIGEST)
        else:
            raise ValueError('Unsupported auth_mode %s' % request.auth_mode)
        userpwd = httputil.encode_username_password(request.auth_username, request.auth_password)
        curl.setopt(pycurl.USERPWD, userpwd)
        curl_log.debug('%s %s (username: %r)', request.method, request.url, request.auth_username)
    else:
        curl.unsetopt(pycurl.USERPWD)
        curl_log.debug('%s %s', request.method, request.url)
    if request.client_cert is not None:
        curl.setopt(pycurl.SSLCERT, request.client_cert)
    if request.client_key is not None:
        curl.setopt(pycurl.SSLKEY, request.client_key)
    if request.ssl_options is not None:
        raise ValueError('ssl_options not supported in curl_httpclient')
    if threading.active_count() > 1:
        curl.setopt(pycurl.NOSIGNAL, 1)
    if request.prepare_curl_callback is not None:
        request.prepare_curl_callback(curl)