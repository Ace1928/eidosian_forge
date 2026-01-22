import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
class _TopLevelStatementsVisitor(Visitor[None]):
    """
    A node visitor that converts unchanged top-level statements to
    STANDALONE_COMMENT.

    This is used in addition to _convert_unchanged_line_by_line, to
    speed up formatting when there are unchanged top-level
    classes/functions/statements.
    """

    def __init__(self, lines_set: Set[int]):
        self._lines_set = lines_set

    def visit_simple_stmt(self, node: Node) -> Iterator[None]:
        yield from []
        newline_leaf = last_leaf(node)
        if not newline_leaf:
            return
        assert newline_leaf.type == NEWLINE, f'Unexpectedly found leaf.type={newline_leaf.type}'
        ancestor = furthest_ancestor_with_last_leaf(newline_leaf)
        if not _get_line_range(ancestor).intersection(self._lines_set):
            _convert_node_to_standalone_comment(ancestor)

    def visit_suite(self, node: Node) -> Iterator[None]:
        yield from []
        if _contains_standalone_comment(node):
            return
        semantic_parent = node.parent
        if semantic_parent is not None:
            if semantic_parent.prev_sibling is not None and semantic_parent.prev_sibling.type == ASYNC:
                semantic_parent = semantic_parent.parent
        if semantic_parent is not None and (not _get_line_range(semantic_parent).intersection(self._lines_set)):
            _convert_node_to_standalone_comment(semantic_parent)