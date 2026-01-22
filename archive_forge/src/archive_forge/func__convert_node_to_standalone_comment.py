import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _convert_node_to_standalone_comment(node: LN) -> None:
    """Convert node to STANDALONE_COMMENT by modifying the tree inline."""
    parent = node.parent
    if not parent:
        return
    first = first_leaf(node)
    last = last_leaf(node)
    if not first or not last:
        return
    if first is last:
        return
    prefix = first.prefix
    first.prefix = ''
    index = node.remove()
    if index is not None:
        value = str(node)[:-1]
        parent.insert_child(index, Leaf(STANDALONE_COMMENT, value, prefix=prefix, fmt_pass_converted_first_leaf=first))