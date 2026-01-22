import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _get_line_range(node_or_nodes: Union[LN, List[LN]]) -> Set[int]:
    """Returns the line range of this node or list of nodes."""
    if isinstance(node_or_nodes, list):
        nodes = node_or_nodes
        if not nodes:
            return set()
        first = first_leaf(nodes[0])
        last = last_leaf(nodes[-1])
        if first and last:
            line_start = first.lineno
            line_end = _leaf_line_end(last)
            return set(range(line_start, line_end + 1))
        else:
            return set()
    else:
        node = node_or_nodes
        if isinstance(node, Leaf):
            return set(range(node.lineno, _leaf_line_end(node) + 1))
        else:
            first = first_leaf(node)
            last = last_leaf(node)
            if first and last:
                return set(range(first.lineno, _leaf_line_end(last) + 1))
            else:
                return set()