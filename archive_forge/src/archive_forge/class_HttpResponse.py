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
class HttpResponse(HttpResponseBase):
    """
    An HTTP response class with a string as content.

    This content can be read, appended to, or replaced.
    """
    streaming = False

    def __init__(self, content=b'', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content = content

    def __repr__(self):
        return '<%(cls)s status_code=%(status_code)d%(content_type)s>' % {'cls': self.__class__.__name__, 'status_code': self.status_code, 'content_type': self._content_type_for_repr}

    def serialize(self):
        """Full HTTP message, including headers, as a bytestring."""
        return self.serialize_headers() + b'\r\n\r\n' + self.content
    __bytes__ = serialize

    @property
    def content(self):
        return b''.join(self._container)

    @content.setter
    def content(self, value):
        if hasattr(value, '__iter__') and (not isinstance(value, (bytes, memoryview, str))):
            content = b''.join((self.make_bytes(chunk) for chunk in value))
            if hasattr(value, 'close'):
                try:
                    value.close()
                except Exception:
                    pass
        else:
            content = self.make_bytes(value)
        self._container = [content]

    def __iter__(self):
        return iter(self._container)

    def write(self, content):
        self._container.append(self.make_bytes(content))

    def tell(self):
        return len(self.content)

    def getvalue(self):
        return self.content

    def writable(self):
        return True

    def writelines(self, lines):
        for line in lines:
            self.write(line)