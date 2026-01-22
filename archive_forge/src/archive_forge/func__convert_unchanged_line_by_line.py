import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _convert_unchanged_line_by_line(node: Node, lines_set: Set[int]) -> None:
    """Converts unchanged to STANDALONE_COMMENT line by line."""
    for leaf in node.leaves():
        if leaf.type != NEWLINE:
            continue
        if leaf.parent and leaf.parent.type == syms.match_stmt:
            nodes_to_ignore: List[LN] = []
            prev_sibling = leaf.prev_sibling
            while prev_sibling:
                nodes_to_ignore.insert(0, prev_sibling)
                prev_sibling = prev_sibling.prev_sibling
            if not _get_line_range(nodes_to_ignore).intersection(lines_set):
                _convert_nodes_to_standalone_comment(nodes_to_ignore, newline=leaf)
        elif leaf.parent and leaf.parent.type == syms.suite:
            parent_sibling = leaf.parent.prev_sibling
            nodes_to_ignore = []
            while parent_sibling and (not parent_sibling.type == syms.suite):
                nodes_to_ignore.insert(0, parent_sibling)
                parent_sibling = parent_sibling.prev_sibling
            grandparent = leaf.parent.parent
            if grandparent is not None and grandparent.prev_sibling is not None and (grandparent.prev_sibling.type == ASYNC):
                nodes_to_ignore.insert(0, grandparent.prev_sibling)
            if not _get_line_range(nodes_to_ignore).intersection(lines_set):
                _convert_nodes_to_standalone_comment(nodes_to_ignore, newline=leaf)
        else:
            ancestor = furthest_ancestor_with_last_leaf(leaf)
            if ancestor.type == syms.decorator and ancestor.parent and (ancestor.parent.type == syms.decorators):
                ancestor = ancestor.parent
            if not _get_line_range(ancestor).intersection(lines_set):
                _convert_node_to_standalone_comment(ancestor)