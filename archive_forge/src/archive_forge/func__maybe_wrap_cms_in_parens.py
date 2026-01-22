import re
import sys
from dataclasses import replace
from enum import Enum, auto
from functools import partial, wraps
from typing import Collection, Iterator, List, Optional, Set, Union, cast
from black.brackets import (
from black.comments import FMT_OFF, generate_comments, list_comments
from black.lines import (
from black.mode import Feature, Mode, Preview
from black.nodes import (
from black.numerics import normalize_numeric_literal
from black.strings import (
from black.trans import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _maybe_wrap_cms_in_parens(node: Node, mode: Mode, features: Collection[Feature]) -> None:
    """When enabled and safe, wrap the multiple context managers in invisible parens.

    It is only safe when `features` contain Feature.PARENTHESIZED_CONTEXT_MANAGERS.
    """
    if Feature.PARENTHESIZED_CONTEXT_MANAGERS not in features or len(node.children) <= 2 or node.children[1].type == syms.atom:
        return
    colon_index: Optional[int] = None
    for i in range(2, len(node.children)):
        if node.children[i].type == token.COLON:
            colon_index = i
            break
    if colon_index is not None:
        lpar = Leaf(token.LPAR, '')
        rpar = Leaf(token.RPAR, '')
        context_managers = node.children[1:colon_index]
        for child in context_managers:
            child.remove()
        new_child = Node(syms.atom, [lpar, Node(syms.testlist_gexp, context_managers), rpar])
        node.insert_child(1, new_child)