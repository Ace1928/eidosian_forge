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
def _parse_uploaders_list_value(token, buffered_iterator):
    value_parts = [token]
    comma_offset = -1
    while comma_offset is not None:
        comma_offset = buffered_iterator.peek_find(_is_comma_token)
        if comma_offset is not None:
            peeked_elements = [value_parts[-1]]
            peeked_elements.extend(buffered_iterator.peek_many(comma_offset - 1))
            comma_was_separator = False
            i = len(peeked_elements) - 1
            while i >= 0:
                token = peeked_elements[i]
                if isinstance(token, Deb822ValueToken):
                    if token.text.endswith('>'):
                        value_parts.extend(buffered_iterator.consume_many(i))
                        assert isinstance(value_parts[-1], Deb822ValueToken) and value_parts[-1].text.endswith('>'), 'Got: ' + str(value_parts)
                        comma_was_separator = True
                    break
                i -= 1
            if comma_was_separator:
                break
            value_parts.extend(buffered_iterator.consume_many(comma_offset))
            assert isinstance(value_parts[-1], Deb822CommaToken)
        else:
            remaining_part = buffered_iterator.peek_buffer()
            consume_elements = len(remaining_part)
            value_parts.extend(remaining_part)
            while value_parts and (not isinstance(value_parts[-1], Deb822ValueToken)):
                value_parts.pop()
                consume_elements -= 1
            buffered_iterator.consume_many(consume_elements)
    return Deb822ParsedValueElement(value_parts)