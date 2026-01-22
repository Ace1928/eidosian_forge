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
def _parse_str(self, content):
    content_len = len(content)
    biter = BufferingIterator(len_check_iterator(content, self._tokenizer(content), content_len=content_len))
    yield from len_check_iterator(content, self._parse_stream(biter), content_len=content_len)