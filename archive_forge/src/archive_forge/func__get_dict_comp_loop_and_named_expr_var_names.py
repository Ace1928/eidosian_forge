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
def _get_dict_comp_loop_and_named_expr_var_names(self, node: ast.DictComp):
    finder = NamedExprFinder()
    for gen in node.generators:
        if isinstance(gen.target, ast.Name):
            yield gen.target.id
        elif isinstance(gen.target, ast.Tuple):
            yield from self._get_names_from_tuple(gen.target)
        finder.visit(gen.ifs)
    yield from finder.names.keys()