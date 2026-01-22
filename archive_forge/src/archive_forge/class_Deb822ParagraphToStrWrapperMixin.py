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
class Deb822ParagraphToStrWrapperMixin(AutoResolvingMixin[str], ABC):

    @property
    def _auto_map_initial_line_whitespace(self):
        return True

    @property
    def _discard_comments_on_read(self):
        return True

    @property
    def _auto_map_final_newline_in_multiline_values(self):
        return True

    @property
    def _preserve_field_comments_on_field_updates(self):
        return True

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

    def __setitem__(self, item, value):
        keep_comments = self._preserve_field_comments_on_field_updates
        comment = None
        if keep_comments and self._auto_resolve_ambiguous_fields:
            keep_comments = None
            key_lookup = item
            if isinstance(item, str):
                key_lookup = (item, 0)
            orig_kvpair = self._paragraph.get_kvpair_element(key_lookup, use_get=True)
            if orig_kvpair is not None:
                comment = orig_kvpair.comment_element
        if self._auto_map_initial_line_whitespace:
            try:
                idx = value.index('\n')
            except ValueError:
                idx = -1
            if idx == -1 or idx == len(value):
                self._paragraph.set_field_to_simple_value(item, value.strip(), preserve_original_field_comment=keep_comments, field_comment=comment)
                return
            first_line, rest = value.split('\n', 1)
            if first_line and first_line[:1] not in ('\t', ' '):
                value = ''.join((' ', first_line.strip(), '\n', rest))
            else:
                value = ''.join((first_line, '\n', rest))
        if not value.endswith('\n'):
            if not self._auto_map_final_newline_in_multiline_values:
                raise ValueError('Values must end with a newline (or be single line values and use the auto whitespace mapping feature)')
            value += '\n'
        self._paragraph.set_field_from_raw_string(item, value, preserve_original_field_comment=keep_comments, field_comment=comment)

    def _interpret_value(self, key, value):
        return self._convert_value_to_str(value)