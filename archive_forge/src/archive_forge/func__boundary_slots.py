from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def _boundary_slots(self, edge, side):
    """
        Assume that the marked subFatGraph has been embedded in the
        plane.  This generator starts at a marked FatEdge and walks
        around one of its adjacent boundary curves (left=-1, right=1),
        yielding all of the pairs (v, s) where s is a slot of the
        vertex v which lies on the specified boundary curve, or
        (v, None) if none of the slots at v lie on the curve.  (To
        extend the embedding over an unmarked arc, the ending slots of
        both ends of the arc must lie on the same boundary curve.
        Flipping may be needed to arrange this.)
        """
    if not edge.marked:
        raise ValueError('Must begin at a marked edge.')
    first_vertex = vertex = edge[1]
    while True:
        end = 0 if edge[0] is vertex else 1
        slot = edge.slots[end]
        for k in range(3):
            slot += side
            interior_edge = self(vertex)[slot]
            if not interior_edge.marked:
                yield (vertex, slot % 4)
            else:
                break
        if k == 0:
            yield (vertex, None)
        if edge is interior_edge:
            raise ValueError('Marked subgraph has a dead end.')
        edge = interior_edge
        vertex = edge(vertex)
        if vertex is first_vertex:
            break