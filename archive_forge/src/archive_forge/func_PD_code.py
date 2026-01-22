import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def PD_code(self, KnotTheory=False, min_strand_index=0):
    """
        The planar diagram code for the link.  When reconstructing a link
        from its PD code, it will not change the ordering of the
        components, and will preserve their orientation except
        possibly for components with only two crossings.

        >>> L = Link('L13n11308')
        >>> [len(c) for c in L.link_components]
        [4, 4, 4, 6, 8]
        >>> L_copy = Link(L.PD_code())
        >>> [len(c) for c in L_copy.link_components]
        [4, 4, 4, 6, 8]
        """
    PD = []
    for c in self.crossings:
        PD.append([s + min_strand_index for s in c.strand_labels])
    if KnotTheory:
        PD = 'PD' + repr(PD).replace('[', 'X[')[1:]
    else:
        PD = [tuple(x) for x in PD]
    return PD