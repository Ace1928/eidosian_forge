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
def _is_identifier(arg):
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
        return False
    return re.match('^[A-Za-z_][A-Za-z0-9_]*$', arg.value) is not None