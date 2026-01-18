from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import VERBOSE
from .mcomplex_with_memory import McomplexWithMemory, edge_and_arrow

        Implements the 2 -> 0 move as a composite of 2<->3
        moves. Based on Segerman's PAMS paper and Matveev's book, the
        details are delicate and confusing in the extreme.
        