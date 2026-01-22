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
class AsyncRequestFactory(RequestFactory):
    """
    Class that lets you create mock ASGI-like Request objects for use in
    testing. Usage:

    rf = AsyncRequestFactory()
    get_request = rf.get("/hello/")
    post_request = rf.post("/submit/", {"foo": "bar"})

    Once you have a request object you can pass it to any view function,
    including synchronous ones. The reason we have a separate class here is:
    a) this makes ASGIRequest subclasses, and
    b) AsyncTestClient can subclass it.
    """

    def _base_scope(self, **request):
        """The base scope for a request."""
        scope = {'asgi': {'version': '3.0'}, 'type': 'http', 'http_version': '1.1', 'client': ['127.0.0.1', 0], 'server': ('testserver', '80'), 'scheme': 'http', 'method': 'GET', 'headers': [], **self.defaults, **request}
        scope['headers'].append((b'cookie', b'; '.join(sorted((('%s=%s' % (morsel.key, morsel.coded_value)).encode('ascii') for morsel in self.cookies.values())))))
        return scope

    def request(self, **request):
        """Construct a generic request object."""
        if '_body_file' in request:
            body_file = request.pop('_body_file')
        else:
            body_file = FakePayload('')
        return ASGIRequest(self._base_scope(**request), LimitedStream(body_file, len(body_file)))

    def generic(self, method, path, data='', content_type='application/octet-stream', secure=False, *, headers=None, **extra):
        """Construct an arbitrary HTTP request."""
        parsed = urlparse(str(path))
        data = force_bytes(data, settings.DEFAULT_CHARSET)
        s = {'method': method, 'path': self._get_path(parsed), 'server': ('127.0.0.1', '443' if secure else '80'), 'scheme': 'https' if secure else 'http', 'headers': [(b'host', b'testserver')]}
        if data:
            s['headers'].extend([(b'content-length', str(len(data)).encode('ascii')), (b'content-type', content_type.encode('ascii'))])
            s['_body_file'] = FakePayload(data)
        if (query_string := extra.pop('QUERY_STRING', None)):
            s['query_string'] = query_string
        if headers:
            extra.update(HttpHeaders.to_asgi_names(headers))
        s['headers'] += [(key.lower().encode('ascii'), value.encode('latin1')) for key, value in extra.items()]
        if not s.get('query_string'):
            s['query_string'] = parsed[4]
        return self.request(**s)