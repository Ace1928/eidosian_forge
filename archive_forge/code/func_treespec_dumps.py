import functools
import warnings
from typing import (
import torch
import optree
from optree import PyTreeSpec  # direct import for type annotations
def treespec_dumps(treespec: TreeSpec, protocol: Optional[int]=None) -> str:
    """Serialize a treespec to a JSON string."""
    if not isinstance(treespec, TreeSpec):
        raise TypeError(f'treespec_dumps(spec): Expected `spec` to be instance of TreeSpec but got item of type {type(treespec)}.')
    from ._pytree import tree_structure as _tree_structure, treespec_dumps as _treespec_dumps
    orig_treespec = _tree_structure(tree_unflatten([0] * treespec.num_leaves, treespec))
    return _treespec_dumps(orig_treespec, protocol=protocol)