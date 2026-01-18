from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def top_pairing(snake):
    cs = snake[-1]
    cn = self.adjacent_upwards(snake.final)
    return tuple(sorted([to_index(cs), to_index(cn)]))