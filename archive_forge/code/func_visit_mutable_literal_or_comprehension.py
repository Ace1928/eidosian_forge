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
def visit_mutable_literal_or_comprehension(self, node):
    if self.arg_depth == 1:
        self.errors.append(B006(node.lineno, node.col_offset))
    self.generic_visit(node)