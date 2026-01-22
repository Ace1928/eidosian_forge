import json
import mimetypes
import os
import sys
from copy import copy
from functools import partial
from http import HTTPStatus
from importlib import import_module
from io import BytesIO, IOBase
from urllib.parse import unquote_to_bytes, urljoin, urlparse, urlsplit
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.core.handlers.base import BaseHandler
from django.core.handlers.wsgi import LimitedStream, WSGIRequest
from django.core.serializers.json import DjangoJSONEncoder
from django.core.signals import got_request_exception, request_finished, request_started
from django.db import close_old_connections
from django.http import HttpHeaders, HttpRequest, QueryDict, SimpleCookie
from django.test import signals
from django.test.utils import ContextList
from django.urls import resolve
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
from django.utils.http import urlencode
from django.utils.itercompat import is_iterable
from django.utils.regex_helper import _lazy_re_compile
class RequestFactory:
    """
    Class that lets you create mock Request objects for use in testing.

    Usage:

    rf = RequestFactory()
    get_request = rf.get('/hello/')
    post_request = rf.post('/submit/', {'foo': 'bar'})

    Once you have a request object you can pass it to any view function,
    just as if that view had been hooked up using a URLconf.
    """

    def __init__(self, *, json_encoder=DjangoJSONEncoder, headers=None, **defaults):
        self.json_encoder = json_encoder
        self.defaults = defaults
        self.cookies = SimpleCookie()
        self.errors = BytesIO()
        if headers:
            self.defaults.update(HttpHeaders.to_wsgi_names(headers))

    def _base_environ(self, **request):
        """
        The base environment for a request.
        """
        return {'HTTP_COOKIE': '; '.join(sorted(('%s=%s' % (morsel.key, morsel.coded_value) for morsel in self.cookies.values()))), 'PATH_INFO': '/', 'REMOTE_ADDR': '127.0.0.1', 'REQUEST_METHOD': 'GET', 'SCRIPT_NAME': '', 'SERVER_NAME': 'testserver', 'SERVER_PORT': '80', 'SERVER_PROTOCOL': 'HTTP/1.1', 'wsgi.version': (1, 0), 'wsgi.url_scheme': 'http', 'wsgi.input': FakePayload(b''), 'wsgi.errors': self.errors, 'wsgi.multiprocess': True, 'wsgi.multithread': False, 'wsgi.run_once': False, **self.defaults, **request}

    def request(self, **request):
        """Construct a generic request object."""
        return WSGIRequest(self._base_environ(**request))

    def _encode_data(self, data, content_type):
        if content_type is MULTIPART_CONTENT:
            return encode_multipart(BOUNDARY, data)
        else:
            match = CONTENT_TYPE_RE.match(content_type)
            if match:
                charset = match[1]
            else:
                charset = settings.DEFAULT_CHARSET
            return force_bytes(data, encoding=charset)

    def _encode_json(self, data, content_type):
        """
        Return encoded JSON if data is a dict, list, or tuple and content_type
        is application/json.
        """
        should_encode = JSON_CONTENT_TYPE_RE.match(content_type) and isinstance(data, (dict, list, tuple))
        return json.dumps(data, cls=self.json_encoder) if should_encode else data

    def _get_path(self, parsed):
        path = parsed.path
        if parsed.params:
            path += ';' + parsed.params
        path = unquote_to_bytes(path)
        return path.decode('iso-8859-1')

    def get(self, path, data=None, secure=False, *, headers=None, **extra):
        """Construct a GET request."""
        data = {} if data is None else data
        return self.generic('GET', path, secure=secure, headers=headers, **{'QUERY_STRING': urlencode(data, doseq=True), **extra})

    def post(self, path, data=None, content_type=MULTIPART_CONTENT, secure=False, *, headers=None, **extra):
        """Construct a POST request."""
        data = self._encode_json({} if data is None else data, content_type)
        post_data = self._encode_data(data, content_type)
        return self.generic('POST', path, post_data, content_type, secure=secure, headers=headers, **extra)

    def head(self, path, data=None, secure=False, *, headers=None, **extra):
        """Construct a HEAD request."""
        data = {} if data is None else data
        return self.generic('HEAD', path, secure=secure, headers=headers, **{'QUERY_STRING': urlencode(data, doseq=True), **extra})

    def trace(self, path, secure=False, *, headers=None, **extra):
        """Construct a TRACE request."""
        return self.generic('TRACE', path, secure=secure, headers=headers, **extra)

    def options(self, path, data='', content_type='application/octet-stream', secure=False, *, headers=None, **extra):
        """Construct an OPTIONS request."""
        return self.generic('OPTIONS', path, data, content_type, secure=secure, headers=headers, **extra)

    def put(self, path, data='', content_type='application/octet-stream', secure=False, *, headers=None, **extra):
        """Construct a PUT request."""
        data = self._encode_json(data, content_type)
        return self.generic('PUT', path, data, content_type, secure=secure, headers=headers, **extra)

    def patch(self, path, data='', content_type='application/octet-stream', secure=False, *, headers=None, **extra):
        """Construct a PATCH request."""
        data = self._encode_json(data, content_type)
        return self.generic('PATCH', path, data, content_type, secure=secure, headers=headers, **extra)

    def delete(self, path, data='', content_type='application/octet-stream', secure=False, *, headers=None, **extra):
        """Construct a DELETE request."""
        data = self._encode_json(data, content_type)
        return self.generic('DELETE', path, data, content_type, secure=secure, headers=headers, **extra)

    def generic(self, method, path, data='', content_type='application/octet-stream', secure=False, *, headers=None, **extra):
        """Construct an arbitrary HTTP request."""
        parsed = urlparse(str(path))
        data = force_bytes(data, settings.DEFAULT_CHARSET)
        r = {'PATH_INFO': self._get_path(parsed), 'REQUEST_METHOD': method, 'SERVER_PORT': '443' if secure else '80', 'wsgi.url_scheme': 'https' if secure else 'http'}
        if data:
            r.update({'CONTENT_LENGTH': str(len(data)), 'CONTENT_TYPE': content_type, 'wsgi.input': FakePayload(data)})
        if headers:
            extra.update(HttpHeaders.to_wsgi_names(headers))
        r.update(extra)
        if not r.get('QUERY_STRING'):
            query_string = parsed[4].encode().decode('iso-8859-1')
            r['QUERY_STRING'] = query_string
        return self.request(**r)