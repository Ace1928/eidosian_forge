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
def _base_scope(self, **request):
    """The base scope for a request."""
    scope = {'asgi': {'version': '3.0'}, 'type': 'http', 'http_version': '1.1', 'client': ['127.0.0.1', 0], 'server': ('testserver', '80'), 'scheme': 'http', 'method': 'GET', 'headers': [], **self.defaults, **request}
    scope['headers'].append((b'cookie', b'; '.join(sorted((('%s=%s' % (morsel.key, morsel.coded_value)).encode('ascii') for morsel in self.cookies.values())))))
    return scope