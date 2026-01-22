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
def _abort_on_error_tokens(sequence):
    for token in sequence:
        if isinstance(token, Deb822ErrorToken):
            error_as_text = token.text.replace('\n', '\\n')
            raise SyntaxOrParseError('Syntax or Parse error on the line: "{error_as_text}"'.format(error_as_text=error_as_text))
        yield token