import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
@dataclass(frozen=True)
class GraphNodeMaker:
    """Helper for making GraphNode and keep tracks of the hierarchical
    grouping.
    """
    parent_path: tuple[str, ...]
    'The parent group path.\n    '

    def subgroup(self, name: str):
        """Start a subgroup with the given name.
        """
        cls = type(self)
        return cls(parent_path=(*self.parent_path, name))

    def make_node(self, **kwargs) -> GraphNode:
        """Make a new node
        """
        return GraphNode(**kwargs, parent_regions=self.parent_path)