from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def KLP_dict(self, vertex, indices):
    """
        Return a dict describing this vertex and its neighbors
        in KLP terminology.

        The translation from our convention is as follows::

                    Y                    Y
                    3                    0
                    ^                    ^
                    |                    |
             0 -----+----> 2 X     1 ----+---> 3 X
                    |                    |
                    |                    |
                    1                    2
               not flipped           flipped

        The indices argument is a dict that assigns an integer
        index to each vertex of the graph.
        """
    KLP = {}
    flipped = self.flipped(vertex)
    edges = self(vertex)
    neighbors = self[vertex]
    strands = [self.KLP_strand(vertex, edge) for edge in edges]
    ids = [indices[v] for v in neighbors]
    KLP['sign'] = 'R' if self.sign(vertex) == 1 else 'L'
    slot = 1 if flipped else 0
    KLP['Xbackward_neighbor'] = ids[slot]
    KLP['Xbackward_strand'] = strands[slot]
    slot = 3 if flipped else 2
    KLP['Xforward_neighbor'] = ids[slot]
    KLP['Xforward_strand'] = strands[slot]
    KLP['Xcomponent'] = edges[slot].component
    slot = 2 if flipped else 1
    KLP['Ybackward_neighbor'] = ids[slot]
    KLP['Ybackward_strand'] = strands[slot]
    slot = 0 if flipped else 3
    KLP['Yforward_neighbor'] = ids[slot]
    KLP['Yforward_strand'] = strands[slot]
    KLP['Ycomponent'] = edges[slot].component
    return KLP