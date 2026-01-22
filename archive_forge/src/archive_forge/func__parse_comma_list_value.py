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
def _parse_comma_list_value(token, buffered_iterator):
    comma_offset = buffered_iterator.peek_find(_is_comma_token)
    value_parts = [token]
    if comma_offset is not None:
        value_parts.extend(buffered_iterator.peek_many(comma_offset - 1))
    else:
        value_parts.extend(buffered_iterator.peek_buffer())
    while value_parts and (not isinstance(value_parts[-1], Deb822ValueToken)):
        value_parts.pop()
    buffered_iterator.consume_many(len(value_parts) - 1)
    return Deb822ParsedValueElement(value_parts)