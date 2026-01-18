import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def writhe(self):
    """
        Finds the writhe of a knot.

        >>> K = Link( [(4,1,5,2), (6,4,7,3), (8,5,1,6), (2,8,3,7)] )  # Figure 8 knot
        >>> K.writhe()
        0
        """
    writhe_value = 0
    for i in range(len(self.crossings)):
        writhe_value += self.crossings[i].sign
    return writhe_value