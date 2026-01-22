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
def _missing__While(self, node: ast.While) -> ast.AST | None:
    body_nodes = self.find_non_missing_node(NodeList(node.body))
    if not body_nodes:
        return None
    new_while = ast.While()
    new_while.lineno = body_nodes.lineno
    new_while.test = ast.Name()
    new_while.test.lineno = body_nodes.lineno
    new_while.test.id = 'True'
    assert hasattr(body_nodes, 'body')
    new_while.body = body_nodes.body
    new_while.orelse = []
    return new_while