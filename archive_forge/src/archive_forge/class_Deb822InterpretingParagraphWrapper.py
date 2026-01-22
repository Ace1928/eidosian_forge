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
class Deb822InterpretingParagraphWrapper(AbstractDeb822ParagraphWrapper[T]):

    def __init__(self, paragraph, interpretation, *, auto_resolve_ambiguous_fields=False, discard_comments_on_read=True):
        super().__init__(paragraph, auto_resolve_ambiguous_fields=auto_resolve_ambiguous_fields, discard_comments_on_read=discard_comments_on_read)
        self._interpretation = interpretation

    def _interpret_value(self, key, value):
        return self._interpretation.interpret(value)