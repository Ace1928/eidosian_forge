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
def add_body_arcs(self, body: Sequence[ast.AST], from_start: ArcStart | None=None, prev_starts: set[ArcStart] | None=None) -> set[ArcStart]:
    """Add arcs for the body of a compound statement.

        `body` is the body node.  `from_start` is a single `ArcStart` that can
        be the previous line in flow before this body.  `prev_starts` is a set
        of ArcStarts that can be the previous line.  Only one of them should be
        given.

        Returns a set of ArcStarts, the exits from this body.

        """
    if prev_starts is None:
        assert from_start is not None
        prev_starts = {from_start}
    for body_node in body:
        lineno = self.line_for_node(body_node)
        first_line = self.multiline.get(lineno, lineno)
        if first_line not in self.statements:
            maybe_body_node = self.find_non_missing_node(body_node)
            if maybe_body_node is None:
                continue
            body_node = maybe_body_node
            lineno = self.line_for_node(body_node)
        for prev_start in prev_starts:
            self.add_arc(prev_start.lineno, lineno, prev_start.cause)
        prev_starts = self.add_arcs(body_node)
    return prev_starts