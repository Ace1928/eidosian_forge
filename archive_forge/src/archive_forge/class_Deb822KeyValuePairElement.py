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
class Deb822KeyValuePairElement(Deb822Element):
    __slots__ = ('_comment_element', '_field_token', '_separator_token', '_value_element')

    def __init__(self, comment_element, field_token, separator_token, value_element):
        super().__init__()
        self._comment_element = comment_element
        self._field_token = field_token
        self._separator_token = separator_token
        self._value_element = value_element
        self._init_parent_of_parts()

    @property
    def field_name(self):
        return self.field_token.text

    @property
    def field_token(self):
        return self._field_token

    @property
    def value_element(self):
        return self._value_element

    @value_element.setter
    def value_element(self, new_value):
        self._value_element.clear_parent_if_parent(self)
        self._value_element = new_value
        new_value.parent_element = self

    def interpret_as(self, interpreter, discard_comments_on_read=True):
        return interpreter.interpret(self, discard_comments_on_read=discard_comments_on_read)

    @property
    def comment_element(self):
        return self._comment_element

    @comment_element.setter
    def comment_element(self, value):
        if value is not None:
            if not value[-1].text.endswith('\n'):
                raise ValueError('Field comments must end with a newline')
        if self._comment_element:
            self._comment_element.clear_parent_if_parent(self)
        if value is not None:
            value.parent_element = self
        self._comment_element = value

    def iter_parts(self):
        if self._comment_element:
            yield self._comment_element
        yield self._field_token
        yield self._separator_token
        yield self._value_element