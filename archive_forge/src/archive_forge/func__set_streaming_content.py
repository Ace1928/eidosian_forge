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
def _set_streaming_content(self, value):
    if not hasattr(value, 'read'):
        self.file_to_stream = None
        return super()._set_streaming_content(value)
    self.file_to_stream = filelike = value
    if hasattr(filelike, 'close'):
        self._resource_closers.append(filelike.close)
    value = iter(lambda: filelike.read(self.block_size), b'')
    self.set_headers(filelike)
    super()._set_streaming_content(value)