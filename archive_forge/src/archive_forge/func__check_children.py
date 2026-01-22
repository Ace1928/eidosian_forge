from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
@staticmethod
def _check_children(children: Mapping[str, Tree]) -> None:
    """Check children for correct types and for any duplicates."""
    if not is_dict_like(children):
        raise TypeError('children must be a dict-like mapping from names to node objects')
    seen = set()
    for name, child in children.items():
        if not isinstance(child, TreeNode):
            raise TypeError(f'Cannot add object {name}. It is of type {type(child)}, but can only add children of type DataTree')
        childid = id(child)
        if childid not in seen:
            seen.add(childid)
        else:
            raise InvalidTreeError(f'Cannot add same node {name} multiple times as different children.')