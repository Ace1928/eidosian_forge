import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
class HTTPGitRequest:
    """Class encapsulating the state of a single git HTTP request.

    Attributes:
      environ: the WSGI environment for the request.
    """

    def __init__(self, environ, start_response, dumb: bool=False, handlers=None) -> None:
        self.environ = environ
        self.dumb = dumb
        self.handlers = handlers
        self._start_response = start_response
        self._cache_headers: List[Tuple[str, str]] = []
        self._headers: List[Tuple[str, str]] = []

    def add_header(self, name, value):
        """Add a header to the response."""
        self._headers.append((name, value))

    def respond(self, status: str=HTTP_OK, content_type: Optional[str]=None, headers: Optional[List[Tuple[str, str]]]=None):
        """Begin a response with the given status and other headers."""
        if headers:
            self._headers.extend(headers)
        if content_type:
            self._headers.append(('Content-Type', content_type))
        self._headers.extend(self._cache_headers)
        return self._start_response(status, self._headers)

    def not_found(self, message: str) -> bytes:
        """Begin a HTTP 404 response and return the text of a message."""
        self._cache_headers = []
        logger.info('Not found: %s', message)
        self.respond(HTTP_NOT_FOUND, 'text/plain')
        return message.encode('ascii')

    def forbidden(self, message: str) -> bytes:
        """Begin a HTTP 403 response and return the text of a message."""
        self._cache_headers = []
        logger.info('Forbidden: %s', message)
        self.respond(HTTP_FORBIDDEN, 'text/plain')
        return message.encode('ascii')

    def error(self, message: str) -> bytes:
        """Begin a HTTP 500 response and return the text of a message."""
        self._cache_headers = []
        logger.error('Error: %s', message)
        self.respond(HTTP_ERROR, 'text/plain')
        return message.encode('ascii')

    def nocache(self) -> None:
        """Set the response to never be cached by the client."""
        self._cache_headers = NO_CACHE_HEADERS

    def cache_forever(self) -> None:
        """Set the response to be cached forever by the client."""
        self._cache_headers = cache_forever_headers()