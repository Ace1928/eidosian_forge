import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
class AWSResponse:
    """A data class representing an HTTP response.

    This class was originally inspired by requests.models.Response, but has
    been boiled down to meet the specific use cases in botocore. This has
    effectively been reduced to a named tuple.

    :ivar url: The full url.
    :ivar status_code: The status code of the HTTP response.
    :ivar headers: The HTTP headers received.
    :ivar body: The HTTP response body.
    """

    def __init__(self, url, status_code, headers, raw):
        self.url = url
        self.status_code = status_code
        self.headers = HeadersDict(headers)
        self.raw = raw
        self._content = None

    @property
    def content(self):
        """Content of the response as bytes."""
        if self._content is None:
            self._content = b''.join(self.raw.stream()) or b''
        return self._content

    @property
    def text(self):
        """Content of the response as a proper text type.

        Uses the encoding type provided in the reponse headers to decode the
        response content into a proper text type. If the encoding is not
        present in the headers, UTF-8 is used as a default.
        """
        encoding = botocore.utils.get_encoding_from_headers(self.headers)
        if encoding:
            return self.content.decode(encoding)
        else:
            return self.content.decode('utf-8')