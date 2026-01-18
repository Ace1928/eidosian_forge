import sys
from typing import (
from mypy_extensions import mypyc_attr
from black.cache import CACHE_DIR
from black.mode import Mode, Preview
from black.strings import get_string_prefix, has_triple_quotes
from blib2to3 import pygram
from blib2to3.pgen2 import token
from blib2to3.pytree import NL, Leaf, Node, type_repr
def unwrap_singleton_parenthesis(node: LN) -> Optional[LN]:
    """Returns `wrapped` if `node` is of the shape ( wrapped ).

    Parenthesis can be optional. Returns None otherwise"""
    if len(node.children) != 3:
        return None
    lpar, wrapped, rpar = node.children
    if not (lpar.type == token.LPAR and rpar.type == token.RPAR):
        return None
    return wrapped