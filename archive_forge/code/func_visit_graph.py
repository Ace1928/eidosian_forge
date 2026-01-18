import abc
from typing import Any
from dataclasses import dataclass, replace, field
from contextlib import contextmanager
from collections import defaultdict
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.core.datastructures.scfg import SCFG
from .regionpasses import RegionVisitor
from .bc2rvsdg import (
def visit_graph(self, scfg: SCFG, builder):
    """Overriding"""
    toposorted = self._toposort_graph(scfg)
    label: str
    last_label: str | None = None
    for lvl in toposorted:
        for label in lvl:
            builder = self.visit(scfg[label], builder)
            if last_label is not None:
                last_node = scfg[last_label]
                node = scfg[label]
                self._connect_inout_ports(last_node, node, builder)
            last_label = label
    return builder