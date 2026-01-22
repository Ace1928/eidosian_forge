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
def append_comment(self, comment_text):
    tail = self._token_list.tail
    if tail is None or not tail.convert_to_text().endswith('\n'):
        self.append_newline()
    comment_token = Deb822CommentToken(_format_comment(comment_text))
    self._token_list.append(comment_token)