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
def _get_text_positions_tokenless(self, node, padded):
    """
    Version of ``get_text_positions()`` that doesn't use tokens.
    """
    if sys.version_info[:2] < (3, 8):
        raise AssertionError('This method should only be called internally after checking supports_tokenless()')
    if is_module(node):
        return ((1, 0), self._line_numbers.offset_to_line(len(self._text)))
    if getattr(node, 'lineno', None) is None:
        return ((1, 0), (1, 0))
    assert node
    decorators = getattr(node, 'decorator_list', [])
    if not decorators:
        decorators_node = getattr(node, 'decorators', None)
        decorators = getattr(decorators_node, 'nodes', [])
    if decorators:
        start_node = decorators[0]
    else:
        start_node = node
    start_lineno = start_node.lineno
    end_node = last_stmt(node)
    if padded and (start_lineno != end_node.lineno or (start_lineno != node.end_lineno and getattr(node, 'doc_node', None) and is_stmt(node))):
        start_col_offset = 0
    else:
        start_col_offset = self._line_numbers.from_utf8_col(start_lineno, start_node.col_offset)
    start = (start_lineno, start_col_offset)
    end_lineno = cast(int, end_node.end_lineno)
    end_col_offset = cast(int, end_node.end_col_offset)
    end_col_offset = self._line_numbers.from_utf8_col(end_lineno, end_col_offset)
    end = (end_lineno, end_col_offset)
    return (start, end)