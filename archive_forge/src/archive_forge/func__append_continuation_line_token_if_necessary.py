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
def _append_continuation_line_token_if_necessary(self):
    tail = self._token_list.tail
    if tail is not None and tail.convert_to_text().endswith('\n'):
        self._token_list.append(Deb822ValueContinuationToken(self._continuation_line_char))