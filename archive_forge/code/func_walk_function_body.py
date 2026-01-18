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
def walk_function_body(self, node):

    def _loop(parent, node):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return
        yield (parent, node)
        for child in ast.iter_child_nodes(node):
            yield from _loop(node, child)
    for child in node.body:
        yield from _loop(node, child)