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
def _dict_or_lambda_match(LL: List[Leaf]) -> Optional[int]:
    """
        Returns:
            string_idx such that @LL[string_idx] is equal to our target (i.e.
            matched) string, if this line matches the dictionary key assignment
            statement or lambda expression requirements listed in the
            'Requirements' section of this classes' docstring.
                OR
            None, otherwise.
        """
    parent_types = [parent_type(LL[0]), parent_type(LL[0].parent)]
    if syms.dictsetmaker in parent_types or syms.lambdef in parent_types:
        is_valid_index = is_valid_index_factory(LL)
        for i, leaf in enumerate(LL):
            if leaf.type == token.COLON and i < len(LL) - 1:
                idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                if is_valid_index(idx) and LL[idx].type == token.STRING:
                    string_idx = idx
                    string_parser = StringParser()
                    idx = string_parser.parse(LL, string_idx)
                    if is_valid_index(idx) and LL[idx].type == token.COMMA:
                        idx += 1
                    if not is_valid_index(idx):
                        return string_idx
    return None