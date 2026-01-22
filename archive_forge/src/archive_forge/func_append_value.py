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
def append_value(self, vt):
    value_parts = self._token_list
    if value_parts:
        needs_separator = False
        stype = self._stype
        vtype = self._vtype
        for t in reversed(value_parts):
            if isinstance(t, vtype):
                needs_separator = True
                break
            if isinstance(t, stype):
                break
        if needs_separator:
            self.append_separator()
    else:
        self._token_list.append(Deb822WhitespaceToken(' '))
    self._append_continuation_line_token_if_necessary()
    self._changed = True
    value_parts.append(vt)