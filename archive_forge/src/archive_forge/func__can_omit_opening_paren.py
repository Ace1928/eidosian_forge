import itertools
import math
from dataclasses import dataclass, field
from typing import (
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import (
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _can_omit_opening_paren(line: Line, *, first: Leaf, line_length: int) -> bool:
    """See `can_omit_invisible_parens`."""
    remainder = False
    length = 4 * line.depth
    _index = -1
    for _index, leaf, leaf_length in line.enumerate_with_length():
        if leaf.type in CLOSING_BRACKETS and leaf.opening_bracket is first:
            remainder = True
        if remainder:
            length += leaf_length
            if length > line_length:
                break
            if leaf.type in OPENING_BRACKETS:
                remainder = False
    else:
        if len(line.leaves) == _index + 1:
            return True
    return False