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
def _rangeToOffsetAndSize(self, start, end):
    """
        Convert a start and end from a Range header to an offset and size.

        This method checks that the resulting range overlaps with the resource
        being served (and so has the value of C{getFileSize()} as an indirect
        input).

        Either but not both of start or end can be L{None}:

         - Omitted start means that the end value is actually a start value
           relative to the end of the resource.

         - Omitted end means the end of the resource should be the end of
           the range.

        End is interpreted as inclusive, as per RFC 2616.

        If this range doesn't overlap with any of this resource, C{(0, 0)} is
        returned, which is not otherwise a value return value.

        @param start: The start value from the header, or L{None} if one was
            not present.
        @param end: The end value from the header, or L{None} if one was not
            present.
        @return: C{(offset, size)} where offset is how far into this resource
            this resource the range begins and size is how long the range is,
            or C{(0, 0)} if the range does not overlap this resource.
        """
    size = self.getFileSize()
    if start is None:
        start = size - end
        end = size
    elif end is None:
        end = size
    elif end < size:
        end += 1
    elif end > size:
        end = size
    if start >= size:
        start = end = 0
    return (start, end - start)