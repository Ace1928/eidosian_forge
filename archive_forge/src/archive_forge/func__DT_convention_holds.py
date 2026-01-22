import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _DT_convention_holds(self):
    for C in self.crossings:
        first, second, flip = C.DT_info()
        if (first + second) % 2 != 1:
            return False
    return True