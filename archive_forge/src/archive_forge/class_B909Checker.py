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
class B909Checker(ast.NodeVisitor):
    MUTATING_FUNCTIONS = ('append', 'sort', 'reverse', 'remove', 'clear', 'extend', 'insert', 'pop', 'popitem')

    def __init__(self, name: str):
        self.name = name
        self.mutations = []

    def visit_Delete(self, node: ast.Delete):
        for target in node.targets:
            if isinstance(target, ast.Subscript):
                name = _to_name_str(target.value)
            elif isinstance(target, (ast.Attribute, ast.Name)):
                name = _to_name_str(target)
            else:
                name = ''
                self.generic_visit(target)
            if name == self.name:
                self.mutations.append(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            name = _to_name_str(node.func.value)
            function_object = name
            function_name = node.func.attr
            if function_object == self.name and function_name in self.MUTATING_FUNCTIONS:
                self.mutations.append(node)
        self.generic_visit(node)

    def visit(self, node):
        """Like super-visit but supports iteration over lists."""
        if not isinstance(node, list):
            return super().visit(node)
        for elem in node:
            super().visit(elem)
        return node