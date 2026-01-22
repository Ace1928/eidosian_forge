import asyncio
import logging
import sys
import tempfile
import traceback
from contextlib import aclosing
from asgiref.sync import ThreadSensitiveContext, sync_to_async
from django.conf import settings
from django.core import signals
from django.core.exceptions import RequestAborted, RequestDataTooBig
from django.core.handlers import base
from django.http import (
from django.urls import set_script_prefix
from django.utils.functional import cached_property
class ASGIRequest(HttpRequest):
    """
    Custom request subclass that decodes from an ASGI-standard request dict
    and wraps request body handling.
    """
    body_receive_timeout = 60

    def __init__(self, scope, body_file):
        self.scope = scope
        self._post_parse_error = False
        self._read_started = False
        self.resolver_match = None
        self.script_name = get_script_prefix(scope)
        if self.script_name:
            self.path_info = scope['path'].removeprefix(self.script_name)
        else:
            self.path_info = scope['path']
        if self.script_name:
            self.path = '%s/%s' % (self.script_name.rstrip('/'), self.path_info.replace('/', '', 1))
        else:
            self.path = scope['path']
        self.method = self.scope['method'].upper()
        query_string = self.scope.get('query_string', '')
        if isinstance(query_string, bytes):
            query_string = query_string.decode()
        self.META = {'REQUEST_METHOD': self.method, 'QUERY_STRING': query_string, 'SCRIPT_NAME': self.script_name, 'PATH_INFO': self.path_info, 'wsgi.multithread': True, 'wsgi.multiprocess': True}
        if self.scope.get('client'):
            self.META['REMOTE_ADDR'] = self.scope['client'][0]
            self.META['REMOTE_HOST'] = self.META['REMOTE_ADDR']
            self.META['REMOTE_PORT'] = self.scope['client'][1]
        if self.scope.get('server'):
            self.META['SERVER_NAME'] = self.scope['server'][0]
            self.META['SERVER_PORT'] = str(self.scope['server'][1])
        else:
            self.META['SERVER_NAME'] = 'unknown'
            self.META['SERVER_PORT'] = '0'
        for name, value in self.scope.get('headers', []):
            name = name.decode('latin1')
            if name == 'content-length':
                corrected_name = 'CONTENT_LENGTH'
            elif name == 'content-type':
                corrected_name = 'CONTENT_TYPE'
            else:
                corrected_name = 'HTTP_%s' % name.upper().replace('-', '_')
            value = value.decode('latin1')
            if corrected_name in self.META:
                value = self.META[corrected_name] + ',' + value
            self.META[corrected_name] = value
        self._set_content_type_params(self.META)
        self._stream = body_file
        self.resolver_match = None

    @cached_property
    def GET(self):
        return QueryDict(self.META['QUERY_STRING'])

    def _get_scheme(self):
        return self.scope.get('scheme') or super()._get_scheme()

    def _get_post(self):
        if not hasattr(self, '_post'):
            self._load_post_and_files()
        return self._post

    def _set_post(self, post):
        self._post = post

    def _get_files(self):
        if not hasattr(self, '_files'):
            self._load_post_and_files()
        return self._files
    POST = property(_get_post, _set_post)
    FILES = property(_get_files)

    @cached_property
    def COOKIES(self):
        return parse_cookie(self.META.get('HTTP_COOKIE', ''))

    def close(self):
        super().close()
        self._stream.close()