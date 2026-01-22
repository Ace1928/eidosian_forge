import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
class AWSRequest:
    """Represents the elements of an HTTP request.

    This class was originally inspired by requests.models.Request, but has been
    boiled down to meet the specific use cases in botocore. That being said this
    class (even in requests) is effectively a named-tuple.
    """
    _REQUEST_PREPARER_CLS = AWSRequestPreparer

    def __init__(self, method=None, url=None, headers=None, data=None, params=None, auth_path=None, stream_output=False):
        self._request_preparer = self._REQUEST_PREPARER_CLS()
        params = {} if params is None else params
        self.method = method
        self.url = url
        self.headers = HTTPHeaders()
        self.data = data
        self.params = params
        self.auth_path = auth_path
        self.stream_output = stream_output
        if headers is not None:
            for key, value in headers.items():
                self.headers[key] = value
        self.context = {}

    def prepare(self):
        """Constructs a :class:`AWSPreparedRequest <AWSPreparedRequest>`."""
        return self._request_preparer.prepare(self)

    @property
    def body(self):
        body = self.prepare().body
        if isinstance(body, str):
            body = body.encode('utf-8')
        return body