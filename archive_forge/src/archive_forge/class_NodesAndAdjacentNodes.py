from __future__ import annotations
import logging # isort:skip
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from .expressions import CoordinateTransform
class NodesAndAdjacentNodes(GraphHitTestPolicy):
    """
    With the ``NodesAndAdjacentNodes`` policy, inspection or selection of
    graph nodes will also result in the inspection or selection any nodes that
    are immediately adjacent (connected by a single edge). There is no
    selection or inspection of graph edges, and no indication of which node is
    the tool-selected one from the policy-selected nodes.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)