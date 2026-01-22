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
class ValueReference(Generic[TE]):
    """Reference to a value inside a Deb822 paragraph

    This is useful for cases where want to modify values "in-place" or maybe
    conditionally remove a value after looking at it.

    ValueReferences can be invalidated by various changes or actions performed
    to the underlying provider of the value reference.  As an example, sorting
    a list of values will generally invalidate all ValueReferences related to
    that list.

    The ValueReference will raise validity issues where it detects them but most
    of the time it will not notice.  As a means to this end,  the ValueReference
    will *not* keep a strong reference to the underlying value.  This enables it
    to detect when the container goes out of scope.  However, keep in mind that
    the timeliness of garbage collection is implementation defined (e.g., pypy
    does not use ref-counting).
    """
    __slots__ = ('_node', '_render', '_value_factory', '_removal_handler', '_mutation_notifier')

    def __init__(self, node, render, value_factory, removal_handler, mutation_notifier):
        self._node = weakref.ref(node)
        self._render = render
        self._value_factory = value_factory
        self._removal_handler = removal_handler
        self._mutation_notifier = mutation_notifier

    def _resolve_node(self):
        if self._node is None:
            raise RuntimeError('Cannot use ValueReference after remove()')
        node = self._node()
        if node is None:
            raise RuntimeError('ValueReference is invalid (garbage collected)')
        return node

    @property
    def value(self):
        """Resolve the reference into a str"""
        return self._render(self._resolve_node().value)

    @value.setter
    def value(self, new_value):
        """Update the reference value

        Updating the value via this method will *not* invalidate the reference (or other
        references to the same container).

        This can raise an exception of the new value does not follow the requirements
        for the referenced values.  As an example, values in whitespace separated
        lists cannot contain spaces and would trigger an exception.
        """
        self._resolve_node().value = self._value_factory(new_value)
        if self._mutation_notifier is not None:
            self._mutation_notifier()

    def remove(self):
        """Remove the underlying value

        This will invalidate the ValueReference (and any other ValueReferences pointing
        to that exact value).  The validity of other ValueReferences to that container
        remains unaffected.
        """
        self._removal_handler(cast('LinkedListNode[TokenOrElement]', self._resolve_node()))
        self._node = None