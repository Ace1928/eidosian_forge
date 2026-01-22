import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def _iter_content_parts(self):
    if self._leading_whitespace_token:
        yield self._leading_whitespace_token
    yield from self._value_tokens
    if self._trailing_whitespace_token:
        yield self._trailing_whitespace_token