import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def active_reac_stoichs(self, keys=None):
    return self._stoichs('active_reac_stoich', keys)