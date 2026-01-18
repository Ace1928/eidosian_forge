import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def tree_any(pred: Callable[[Any], bool], tree: PyTree) -> bool:
    flat_args = tree_leaves(tree)
    return any(map(pred, flat_args))