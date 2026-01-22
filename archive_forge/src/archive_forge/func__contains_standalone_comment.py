import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def _contains_standalone_comment(node: LN) -> bool:
    if isinstance(node, Leaf):
        return node.type == STANDALONE_COMMENT
    else:
        for child in node.children:
            if _contains_standalone_comment(child):
                return True
        return False