from __future__ import annotations
import mimetypes
from io import BytesIO
from os import fsdecode
from os import fspath
from .._internal import _plain_int
from .structures import MultiDict
from .. import http
Adds a new file to the dict.  `file` can be a file name or
        a :class:`file`-like or a :class:`FileStorage` object.

        :param name: the name of the field.
        :param file: a filename or :class:`file`-like object
        :param filename: an optional filename
        :param content_type: an optional content type
        