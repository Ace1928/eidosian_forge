from __future__ import annotations
import errno
import itertools
import mimetypes
import os
import time
import warnings
from html import escape
from typing import Any, Callable, Dict, Sequence
from urllib.parse import quote, unquote
from zope.interface import implementer
from incremental import Version
from typing_extensions import Literal
from twisted.internet import abstract, interfaces
from twisted.python import components, filepath, log
from twisted.python.compat import nativeString, networkString
from twisted.python.deprecate import deprecated
from twisted.python.runtime import platformType
from twisted.python.url import URL
from twisted.python.util import InsensitiveDict
from twisted.web import http, resource, server
from twisted.web.util import redirectTo
def _contentRange(self, offset, size):
    """
        Return a string suitable for the value of a Content-Range header for a
        range with the given offset and size.

        The offset and size are not sanity checked in any way.

        @param offset: How far into this resource the range begins.
        @param size: How long the range is.
        @return: The value as appropriate for the value of a Content-Range
            header.
        """
    return networkString('bytes %d-%d/%d' % (offset, offset + size - 1, self.getFileSize()))