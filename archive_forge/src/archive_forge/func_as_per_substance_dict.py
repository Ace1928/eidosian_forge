import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def as_per_substance_dict(self, arr):
    return dict(zip(self.substances.keys(), arr))