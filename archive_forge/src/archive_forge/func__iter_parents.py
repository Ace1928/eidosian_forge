from __future__ import annotations
import sys
from collections.abc import Iterator, Mapping
from pathlib import PurePosixPath
from typing import (
from xarray.core.utils import Frozen, is_dict_like
def _iter_parents(self: Tree) -> Iterator[Tree]:
    """Iterate up the tree, starting from the current node's parent."""
    node: Tree | None = self.parent
    while node is not None:
        yield node
        node = node.parent