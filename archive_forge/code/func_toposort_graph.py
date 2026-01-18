import abc
from collections.abc import Mapping
from typing import TypeVar, Generic
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
def toposort_graph(graph: Mapping[str, BasicBlock]) -> list[list[str]]:
    """Topologically sort the graph returning a list.

    The first element of the list is the source and the last element is the
    sink, according to the direction of the dataflow.
    Each element of the list is a list of nodes at the same topological level.
    """
    incoming_labels = _compute_incoming_labels(graph)
    visited: set[str] = set()
    toposorted: list[list[str]] = []
    while incoming_labels:
        level = []
        for k, vs in incoming_labels.items():
            if not vs - visited:
                level.append(k)
        for k in level:
            del incoming_labels[k]
        visited |= set(level)
        toposorted.append(level)
    return toposorted