from itertools import groupby
import numpy as np
import pandas as pd
import param
from .dimension import Dimensioned, ViewableElement, asdim
from .layout import Composable, Layout, NdLayout
from .ndmapping import NdMapping
from .overlay import CompositeOverlay, NdOverlay, Overlayable
from .spaces import GridSpace, HoloMap
from .tree import AttrTree
from .util import get_param_values
def _reduce_map(self, dimensions, function, reduce_map):
    if dimensions and reduce_map:
        raise Exception('Pass reduced dimensions either as an argument or as part of the kwargs not both.')
    if len(set(reduce_map.values())) > 1:
        raise Exception('Cannot define reduce operations with more than one function at a time.')
    if reduce_map:
        reduce_map = reduce_map.items()
    if dimensions:
        reduce_map = [(d, function) for d in dimensions]
    elif not reduce_map:
        reduce_map = [(d, function) for d in self.kdims]
    reduced = [(self.get_dimension(d, strict=True).name, fn) for d, fn in reduce_map]
    grouped = [(fn, [dim for dim, _ in grp]) for fn, grp in groupby(reduced, lambda x: x[1])]
    return grouped[0]