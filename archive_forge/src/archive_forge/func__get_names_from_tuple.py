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
def _get_names_from_tuple(self, node: ast.Tuple):
    for dim in node.elts:
        if isinstance(dim, ast.Name):
            yield dim.id
        elif isinstance(dim, ast.Tuple):
            yield from self._get_names_from_tuple(dim)