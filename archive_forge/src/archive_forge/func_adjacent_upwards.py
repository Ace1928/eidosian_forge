from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def adjacent_upwards(self, crossing_strand):
    a, b = (crossing_strand.rotate(), crossing_strand.rotate(-1))
    if self.orientations[a] in ['up', 'min']:
        return a
    else:
        assert self.orientations[b] in ['up', 'min']
        return b