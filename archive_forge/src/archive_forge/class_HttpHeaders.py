import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
from django.core.files import uploadhandler
from django.http.multipartparser import (
from django.utils.datastructures import (
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
class HttpHeaders(CaseInsensitiveMapping):
    HTTP_PREFIX = 'HTTP_'
    UNPREFIXED_HEADERS = {'CONTENT_TYPE', 'CONTENT_LENGTH'}

    def __init__(self, environ):
        headers = {}
        for header, value in environ.items():
            name = self.parse_header_name(header)
            if name:
                headers[name] = value
        super().__init__(headers)

    def __getitem__(self, key):
        """Allow header lookup using underscores in place of hyphens."""
        return super().__getitem__(key.replace('_', '-'))

    @classmethod
    def parse_header_name(cls, header):
        if header.startswith(cls.HTTP_PREFIX):
            header = header.removeprefix(cls.HTTP_PREFIX)
        elif header not in cls.UNPREFIXED_HEADERS:
            return None
        return header.replace('_', '-').title()

    @classmethod
    def to_wsgi_name(cls, header):
        header = header.replace('-', '_').upper()
        if header in cls.UNPREFIXED_HEADERS:
            return header
        return f'{cls.HTTP_PREFIX}{header}'

    @classmethod
    def to_asgi_name(cls, header):
        return header.replace('-', '_').upper()

    @classmethod
    def to_wsgi_names(cls, headers):
        return {cls.to_wsgi_name(header_name): value for header_name, value in headers.items()}

    @classmethod
    def to_asgi_names(cls, headers):
        return {cls.to_asgi_name(header_name): value for header_name, value in headers.items()}