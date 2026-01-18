from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def two_cycles(self):
    """
        Find all two cycles and yield them as a pair of CrossingStrands which
        are dual to the edges in the cycle.

        The crossing strands are
        oriented consistently with respect to one of the faces which a
        vertex for the cycle.
        """
    for face0 in self.vertices:
        for dual_edge0 in self.incident(face0):
            face1 = dual_edge0(face0)
            if face0.label < face1.label:
                for dual_edge1 in self.incident(face1):
                    if dual_edge0.label < dual_edge1.label and dual_edge1(face1) == face0:
                        yield (common_element(face0, dual_edge0.interface), common_element(face0, dual_edge1.interface))