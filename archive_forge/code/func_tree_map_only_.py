import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def tree_map_only_(__type_or_types: TypeAny, func: FnAny[Any], tree: PyTree) -> PyTree:
    return tree_map_(map_only(__type_or_types)(func), tree)