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
def visit_tname(self, node: Node) -> Iterator[Line]:
    """
        Add potential parentheses around types in function parameter lists to be made
        into real parentheses in case the type hint is too long to fit on a line
        Examples:
        def foo(a: int, b: float = 7): ...

        ->

        def foo(a: (int), b: (float) = 7): ...
        """
    assert len(node.children) == 3
    if maybe_make_parens_invisible_in_atom(node.children[2], parent=node):
        wrap_in_parentheses(node, node.children[2], visible=False)
    yield from self.visit_default(node)