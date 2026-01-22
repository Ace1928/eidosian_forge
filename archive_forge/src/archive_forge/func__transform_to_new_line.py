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
def _transform_to_new_line(self, line: Line, string_and_rpar_indices: List[int]) -> Line:
    LL = line.leaves
    new_line = line.clone()
    new_line.comments = line.comments.copy()
    previous_idx = -1
    for idx in sorted(string_and_rpar_indices):
        leaf = LL[idx]
        lpar_or_rpar_idx = idx - 1 if leaf.type == token.STRING else idx
        append_leaves(new_line, line, LL[previous_idx + 1:lpar_or_rpar_idx])
        if leaf.type == token.STRING:
            string_leaf = Leaf(token.STRING, LL[idx].value)
            LL[lpar_or_rpar_idx].remove()
            replace_child(LL[idx], string_leaf)
            new_line.append(string_leaf)
            old_comments = new_line.comments.pop(id(LL[idx]), [])
            new_line.comments.setdefault(id(string_leaf), []).extend(old_comments)
        else:
            LL[lpar_or_rpar_idx].remove()
        previous_idx = idx
    append_leaves(new_line, line, LL[idx + 1:])
    return new_line