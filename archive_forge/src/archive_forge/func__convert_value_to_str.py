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
def _convert_value_to_str(self, kvpair_element):
    value_element = kvpair_element.value_element
    value_entries = value_element.value_lines
    if len(value_entries) == 1:
        value_entry = value_entries[0]
        t = value_entry.convert_to_text()
        if self._auto_map_initial_line_whitespace:
            t = t.strip()
        return t
    if self._auto_map_initial_line_whitespace or self._discard_comments_on_read:
        converter = _convert_value_lines_to_lines(value_entries, self._discard_comments_on_read)
        auto_map_space = self._auto_map_initial_line_whitespace
        as_text = ''.join((line.strip() + '\n' if auto_map_space and i == 1 else line for i, line in enumerate(converter, start=1)))
    else:
        as_text = value_element.convert_to_text()
    if self._auto_map_final_newline_in_multiline_values and as_text[-1] == '\n':
        as_text = as_text[:-1]
    return as_text