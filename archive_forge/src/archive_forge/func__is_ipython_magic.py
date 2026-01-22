import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
def _is_ipython_magic(node: ast.expr) -> TypeGuard[ast.Attribute]:
    """Check if attribute is IPython magic.

    Note that the source of the abstract syntax tree
    will already have been processed by IPython's
    TransformerManager().transform_cell.
    """
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and (node.value.func.id == 'get_ipython')