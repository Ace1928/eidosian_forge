import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _pieces(self):
    """
        Auxiliary function used by knot_group. Constructs the strands
        of the knot from under-crossing to under-crossing. Needed for the
        Wirtinger Presentation.
        """
    pieces = []
    for s, x in enumerate(self.crossings):
        y = x
        l = 2
        go = True
        pieces.append([])
        pieces[s].append(x[l])
        while go:
            if y.adjacent[l][1] == 0:
                pieces[s].append(y.adjacent[l][0][0])
                break
            pieces[s].append(y.adjacent[l])
            if y.adjacent[l][1] == 1:
                pieces[s].append(y.adjacent[l][0][3])
            if y.adjacent[l][1] == 3:
                pieces[s].append(y.adjacent[l][0][1])
            lnew = (y.adjacent[l][1] + 2) % 4
            y = y.adjacent[l][0]
            l = lnew
    return pieces