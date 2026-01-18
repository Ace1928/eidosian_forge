import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def vertex_to_KLP(c, v):
    i = v if c.sign == 1 else (v - 1) % 4
    return ['Ybackward', 'Xforward', 'Yforward', 'Xbackward'][i]