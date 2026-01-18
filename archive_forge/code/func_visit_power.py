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
def visit_power(self, node: Node) -> Iterator[Line]:
    for idx, leaf in enumerate(node.children[:-1]):
        next_leaf = node.children[idx + 1]
        if not isinstance(leaf, Leaf):
            continue
        value = leaf.value.lower()
        if leaf.type == token.NUMBER and next_leaf.type == syms.trailer and (next_leaf.children[0].type == token.DOT) and (not value.startswith(('0x', '0b', '0o'))) and ('j' not in value):
            wrap_in_parentheses(node, leaf)
    remove_await_parens(node)
    yield from self.visit_default(node)