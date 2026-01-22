from __future__ import annotations
import ast
import builtins
import itertools
import logging
import math
import re
import sys
import warnings
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from keyword import iskeyword
from typing import Dict, List, Set, Union
import attr
import pycodestyle
class FuntionDefDefaultsVisitor(ast.NodeVisitor):

    def __init__(self, b008_extend_immutable_calls=None):
        self.b008_extend_immutable_calls = b008_extend_immutable_calls or set()
        for node in B006.mutable_literals + B006.mutable_comprehensions:
            setattr(self, f'visit_{node}', self.visit_mutable_literal_or_comprehension)
        self.errors = []
        self.arg_depth = 0
        super().__init__()

    def visit_mutable_literal_or_comprehension(self, node):
        if self.arg_depth == 1:
            self.errors.append(B006(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_Call(self, node):
        call_path = '.'.join(compose_call_path(node.func))
        if call_path in B006.mutable_calls:
            self.errors.append(B006(node.lineno, node.col_offset))
            self.generic_visit(node)
            return
        if call_path in B008.immutable_calls | self.b008_extend_immutable_calls:
            self.generic_visit(node)
            return
        if call_path == 'float' and len(node.args) == 1:
            try:
                value = float(ast.literal_eval(node.args[0]))
            except Exception:
                pass
            else:
                if math.isfinite(value):
                    self.errors.append(B008(node.lineno, node.col_offset))
        else:
            self.errors.append(B008(node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_Lambda(self, node):
        pass

    def visit(self, node):
        """Like super-visit but supports iteration over lists."""
        self.arg_depth += 1
        if isinstance(node, list):
            for elem in node:
                if elem is not None:
                    super().visit(elem)
        else:
            super().visit(node)
        self.arg_depth -= 1