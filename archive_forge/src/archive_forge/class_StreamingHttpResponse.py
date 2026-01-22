import datetime
import io
import json
import mimetypes
import os
import re
import sys
import time
import warnings
from email.header import Header
from http.client import responses
from urllib.parse import urlparse
from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.core import signals, signing
from django.core.exceptions import DisallowedRedirect
from django.core.serializers.json import DjangoJSONEncoder
from django.http.cookie import SimpleCookie
from django.utils import timezone
from django.utils.datastructures import CaseInsensitiveMapping
from django.utils.encoding import iri_to_uri
from django.utils.http import content_disposition_header, http_date
from django.utils.regex_helper import _lazy_re_compile
class StreamingHttpResponse(HttpResponseBase):
    """
    A streaming HTTP response class with an iterator as content.

    This should only be iterated once, when the response is streamed to the
    client. However, it can be appended to or replaced with a new iterator
    that wraps the original content (or yields entirely new content).
    """
    streaming = True

    def __init__(self, streaming_content=(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_content = streaming_content

    def __repr__(self):
        return '<%(cls)s status_code=%(status_code)d%(content_type)s>' % {'cls': self.__class__.__qualname__, 'status_code': self.status_code, 'content_type': self._content_type_for_repr}

    @property
    def content(self):
        raise AttributeError('This %s instance has no `content` attribute. Use `streaming_content` instead.' % self.__class__.__name__)

    @property
    def streaming_content(self):
        if self.is_async:
            _iterator = self._iterator

            async def awrapper():
                async for part in _iterator:
                    yield self.make_bytes(part)
            return awrapper()
        else:
            return map(self.make_bytes, self._iterator)

    @streaming_content.setter
    def streaming_content(self, value):
        self._set_streaming_content(value)

    def _set_streaming_content(self, value):
        try:
            self._iterator = iter(value)
            self.is_async = False
        except TypeError:
            self._iterator = aiter(value)
            self.is_async = True
        if hasattr(value, 'close'):
            self._resource_closers.append(value.close)

    def __iter__(self):
        try:
            return iter(self.streaming_content)
        except TypeError:
            warnings.warn('StreamingHttpResponse must consume asynchronous iterators in order to serve them synchronously. Use a synchronous iterator instead.', Warning)

            async def to_list(_iterator):
                as_list = []
                async for chunk in _iterator:
                    as_list.append(chunk)
                return as_list
            return map(self.make_bytes, iter(async_to_sync(to_list)(self._iterator)))

    async def __aiter__(self):
        try:
            async for part in self.streaming_content:
                yield part
        except TypeError:
            warnings.warn('StreamingHttpResponse must consume synchronous iterators in order to serve them asynchronously. Use an asynchronous iterator instead.', Warning)
            for part in await sync_to_async(list)(self.streaming_content):
                yield part

    def getvalue(self):
        return b''.join(self.streaming_content)