from __future__ import annotations
import dataclasses
import mimetypes
import sys
import typing as t
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from itertools import chain
from random import random
from tempfile import TemporaryFile
from time import time
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from ._internal import _get_environ
from ._internal import _wsgi_decoding_dance
from ._internal import _wsgi_encoding_dance
from .datastructures import Authorization
from .datastructures import CallbackDict
from .datastructures import CombinedMultiDict
from .datastructures import EnvironHeaders
from .datastructures import FileMultiDict
from .datastructures import Headers
from .datastructures import MultiDict
from .http import dump_cookie
from .http import dump_options_header
from .http import parse_cookie
from .http import parse_date
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartEncoder
from .sansio.multipart import Preamble
from .urls import _urlencode
from .urls import iri_to_uri
from .utils import cached_property
from .utils import get_content_type
from .wrappers.request import Request
from .wrappers.response import Response
from .wsgi import ClosingIterator
from .wsgi import get_current_url
class TestResponse(Response):
    """:class:`~werkzeug.wrappers.Response` subclass that provides extra
    information about requests made with the test :class:`Client`.

    Test client requests will always return an instance of this class.
    If a custom response class is passed to the client, it is
    subclassed along with this to support test information.

    If the test request included large files, or if the application is
    serving a file, call :meth:`close` to close any open files and
    prevent Python showing a ``ResourceWarning``.

    .. versionchanged:: 2.2
        Set the ``default_mimetype`` to None to prevent a mimetype being
        assumed if missing.

    .. versionchanged:: 2.1
        Response instances cannot be treated as tuples.

    .. versionadded:: 2.0
        Test client methods always return instances of this class.
    """
    default_mimetype = None
    request: Request
    'A request object with the environ used to make the request that\n    resulted in this response.\n    '
    history: tuple[TestResponse, ...]
    'A list of intermediate responses. Populated when the test request\n    is made with ``follow_redirects`` enabled.\n    '
    __test__ = False

    def __init__(self, response: t.Iterable[bytes], status: str, headers: Headers, request: Request, history: tuple[TestResponse]=(), **kwargs: t.Any) -> None:
        super().__init__(response, status, headers, **kwargs)
        self.request = request
        self.history = history
        self._compat_tuple = (response, status, headers)

    @cached_property
    def text(self) -> str:
        """The response data as text. A shortcut for
        ``response.get_data(as_text=True)``.

        .. versionadded:: 2.1
        """
        return self.get_data(as_text=True)