import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _generate_ignored_nodes_from_fmt_skip(leaf: Leaf, comment: ProtoComment) -> Iterator[LN]:
    """Generate all leaves that should be ignored by the `# fmt: skip` from `leaf`."""
    prev_sibling = leaf.prev_sibling
    parent = leaf.parent
    comments = list_comments(leaf.prefix, is_endmarker=False)
    if not comments or comment.value != comments[0].value:
        return
    if prev_sibling is not None:
        leaf.prefix = ''
        siblings = [prev_sibling]
        while '\n' not in prev_sibling.prefix and prev_sibling.prev_sibling is not None:
            prev_sibling = prev_sibling.prev_sibling
            siblings.insert(0, prev_sibling)
        yield from siblings
    elif parent is not None and parent.type == syms.suite and (leaf.type == token.NEWLINE):
        leaf.prefix = ''
        ignored_nodes: List[LN] = []
        parent_sibling = parent.prev_sibling
        while parent_sibling is not None and parent_sibling.type != syms.suite:
            ignored_nodes.insert(0, parent_sibling)
            parent_sibling = parent_sibling.prev_sibling
        grandparent = parent.parent
        if grandparent is not None and grandparent.prev_sibling is not None and (grandparent.prev_sibling.type == token.ASYNC):
            ignored_nodes.insert(0, grandparent.prev_sibling)
        yield from iter(ignored_nodes)