import abc
import ast
import bisect
import sys
import token
from ast import Module
from typing import Iterable, Iterator, List, Optional, Tuple, Any, cast, TYPE_CHECKING
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from .line_numbers import LineNumbers
from .util import (
class ASTTextBase(six.with_metaclass(abc.ABCMeta, object)):

    def __init__(self, source_text, filename):
        self._filename = filename
        source_text = six.ensure_text(source_text)
        self._text = source_text
        self._line_numbers = LineNumbers(source_text)

    @abc.abstractmethod
    def get_text_positions(self, node, padded):
        """
    Returns two ``(lineno, col_offset)`` tuples for the start and end of the given node.
    If the positions can't be determined, or the nodes don't correspond to any particular text,
    returns ``(1, 0)`` for both.

    ``padded`` corresponds to the ``padded`` argument to ``ast.get_source_segment()``.
    This means that if ``padded`` is True, the start position will be adjusted to include
    leading whitespace if ``node`` is a multiline statement.
    """
        raise NotImplementedError

    def get_text_range(self, node, padded=True):
        """
    Returns the (startpos, endpos) positions in source text corresponding to the given node.
    Returns (0, 0) for nodes (like `Load`) that don't correspond to any particular text.

    See ``get_text_positions()`` for details on the ``padded`` argument.
    """
        start, end = self.get_text_positions(node, padded)
        return (self._line_numbers.line_to_offset(*start), self._line_numbers.line_to_offset(*end))

    def get_text(self, node, padded=True):
        """
    Returns the text corresponding to the given node.
    Returns '' for nodes (like `Load`) that don't correspond to any particular text.

    See ``get_text_positions()`` for details on the ``padded`` argument.
    """
        start, end = self.get_text_range(node, padded)
        return self._text[start:end]