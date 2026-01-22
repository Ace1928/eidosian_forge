import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def _broadcast_to_and_flatten(tree: PyTree, treespec: TreeSpec) -> Optional[List[Any]]:
    assert isinstance(treespec, TreeSpec)
    full_tree = tree_unflatten([0] * treespec.num_leaves, treespec)
    try:
        return broadcast_prefix(tree, full_tree)
    except ValueError:
        return None