import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import (
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.rusty import Err, Ok, Result
from black.strings import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
@staticmethod
def _else_match(LL: List[Leaf]) -> Optional[int]:
    """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the ternary expression
            requirements listed in the 'Requirements' section of this classes'
            docstring.
                OR
            None, otherwise.
        """
    if parent_type(LL[0]) == syms.test and LL[0].type == token.NAME and (LL[0].value == 'else'):
        is_valid_index = is_valid_index_factory(LL)
        idx = 2 if is_valid_index(1) and is_empty_par(LL[1]) else 1
        if is_valid_index(idx) and LL[idx].type == token.STRING:
            return idx
    return None