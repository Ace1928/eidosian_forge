from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class NodesAndLinkedEdges(GraphHitTestPolicy):
    """
    With the ``NodesAndLinkedEdges`` policy, inspection or selection of graph
    nodes will result in the inspection or selection of the node and of the
    linked graph edges. There is no direct selection or inspection of graph
    edges.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)