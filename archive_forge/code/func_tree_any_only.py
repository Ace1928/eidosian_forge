import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def tree_any_only(__type_or_types: TypeAny, pred: FnAny[bool], tree: PyTree) -> bool:
    flat_args = tree_leaves(tree)
    return any((pred(x) for x in flat_args if isinstance(x, __type_or_types)))