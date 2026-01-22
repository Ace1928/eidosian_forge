import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def _render_group(self, renderer, group: GraphGroup):
    """Recursively rendering the hierarchical groups
        """
    for k, subgroup in group.subgroups.items():
        with renderer.render_cluster(k) as subrenderer:
            self._render_group(subrenderer, subgroup)
    for k in group.nodes:
        node = self._nodes[k]
        renderer.render_node(k, node)