from __future__ import annotations
import ast
import collections
import os
import re
import sys
import token
import tokenize
from dataclasses import dataclass
from types import CodeType
from typing import (
from coverage import env
from coverage.bytecode import code_objects
from coverage.debug import short_stack
from coverage.exceptions import NoSource, NotPython
from coverage.misc import join_regex, nice_pair
from coverage.phystokens import generate_tokens
from coverage.types import TArc, TLineNo
def _handle__While(self, node: ast.While) -> set[ArcStart]:
    start = to_top = self.line_for_node(node.test)
    constant_test = self.is_constant_expr(node.test)
    top_is_body0 = False
    if constant_test:
        top_is_body0 = True
    if env.PYBEHAVIOR.keep_constant_test:
        top_is_body0 = False
    if top_is_body0:
        to_top = self.line_for_node(node.body[0])
    self.block_stack.append(LoopBlock(start=to_top))
    from_start = ArcStart(start, cause='the condition on line {lineno} was never true')
    exits = self.add_body_arcs(node.body, from_start=from_start)
    for xit in exits:
        self.add_arc(xit.lineno, to_top, xit.cause)
    exits = set()
    my_block = self.block_stack.pop()
    assert isinstance(my_block, LoopBlock)
    exits.update(my_block.break_exits)
    from_start = ArcStart(start, cause='the condition on line {lineno} was never false')
    if node.orelse:
        else_exits = self.add_body_arcs(node.orelse, from_start=from_start)
        exits |= else_exits
    elif not constant_test:
        exits.add(from_start)
    return exits