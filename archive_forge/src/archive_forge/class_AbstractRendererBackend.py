import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
class AbstractRendererBackend(abc.ABC):
    """Base class for all renderer backend.
    """

    @abc.abstractmethod
    def render_node(self, k: str, node: GraphNode):
        ...

    @abc.abstractmethod
    def render_edge(self, edge: GraphEdge):
        ...

    @contextmanager
    @abc.abstractmethod
    def render_cluster(self, name: str):
        ...