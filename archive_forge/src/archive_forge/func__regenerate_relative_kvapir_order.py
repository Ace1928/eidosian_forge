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
def _regenerate_relative_kvapir_order(self, field_name):
    nodes = []
    for node in self._kvpair_order.iter_nodes():
        if node.value.field_name == field_name:
            nodes.append(node)
    self._kvpair_elements[field_name] = nodes