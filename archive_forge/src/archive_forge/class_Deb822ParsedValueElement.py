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
class Deb822ParsedValueElement(Deb822Element):
    __slots__ = ('_text_cached', '_text_no_comments_cached', '_token_list')

    def __init__(self, tokens):
        super().__init__()
        self._token_list = tokens
        self._init_parent_of_parts()
        if not isinstance(tokens[0], Deb822ValueToken) or not isinstance(tokens[-1], Deb822ValueToken):
            raise ValueError(self.__class__.__name__ + ' MUST start and end on a Deb822ValueToken')
        if len(tokens) == 1:
            token = tokens[0]
            self._text_cached = token.text
            self._text_no_comments_cached = token.text
        else:
            self._text_cached = None
            self._text_no_comments_cached = None

    def convert_to_text(self):
        if self._text_no_comments_cached is None:
            self._text_no_comments_cached = super().convert_to_text()
        return self._text_no_comments_cached

    def convert_to_text_without_comments(self):
        if self._text_no_comments_cached is None:
            self._text_no_comments_cached = ''.join((t.text for t in self.iter_tokens() if not t.is_comment))
        return self._text_no_comments_cached

    def iter_parts(self):
        yield from self._token_list