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
def _handle_decorated(self, node: ast.FunctionDef) -> set[ArcStart]:
    """Add arcs for things that can be decorated (classes and functions)."""
    main_line: TLineNo = node.lineno
    last: TLineNo | None = node.lineno
    decs = node.decorator_list
    if decs:
        last = None
        for dec_node in decs:
            dec_start = self.line_for_node(dec_node)
            if last is not None and dec_start != last:
                self.add_arc(last, dec_start)
            last = dec_start
        assert last is not None
        self.add_arc(last, main_line)
        last = main_line
        if env.PYBEHAVIOR.trace_decorator_line_again:
            for top, bot in zip(decs, decs[1:]):
                self.add_arc(self.line_for_node(bot), self.line_for_node(top))
            self.add_arc(self.line_for_node(decs[0]), main_line)
            self.add_arc(main_line, self.line_for_node(decs[-1]))
        if node.body:
            body_start = self.line_for_node(node.body[0])
            body_start = self.multiline.get(body_start, body_start)
    assert last is not None
    return {ArcStart(last)}