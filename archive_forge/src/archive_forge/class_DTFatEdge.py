from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
class DTFatEdge(FatEdge):
    """
    A fat edge which can be marked and belongs to a link component.
    """

    def __init__(self, x, y, twists=0, component=0):
        FatEdge.__init__(self, x, y, twists)
        self.marked = False
        self.component = component

    def PD_index(self):
        """
        The labelling of vertices when building a DT code also
        determines a labelling of the edges, which is needed
        for generating a PD description of the diagram.
        This method returns the edge label.
        """
        v = self[0]
        if self.slot(v) % 2 == 0:
            return v[0]
        else:
            return v[1]