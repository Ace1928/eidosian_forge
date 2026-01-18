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
def visit_async_stmt(self, node: Node) -> Iterator[Line]:
    """Visit `async def`, `async for`, `async with`."""
    yield from self.line()
    children = iter(node.children)
    for child in children:
        yield from self.visit(child)
        if child.type == token.ASYNC or child.type == STANDALONE_COMMENT:
            break
    internal_stmt = next(children)
    yield from self.visit(internal_stmt)