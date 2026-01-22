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
def _add_dimensions(self, item, dims, constant_keys):
    """
        Recursively descend through an Layout and NdMapping objects
        in order to add the supplied dimension values to all contained
        HoloMaps.
        """
    if isinstance(item, Layout):
        item.fixed = False
    dim_vals = [(dim, val) for dim, val in dims[::-1] if dim not in self.drop]
    if isinstance(item, self.merge_type):
        new_item = item.clone(cdims=constant_keys)
        for dim, val in dim_vals:
            dim = asdim(dim)
            if dim not in new_item.kdims:
                new_item = new_item.add_dimension(dim, 0, val)
    elif isinstance(item, self._nest_order[self.merge_type]):
        if len(dim_vals):
            dimensions, key = zip(*dim_vals)
            new_item = self.merge_type({key: item}, kdims=list(dimensions), cdims=constant_keys)
        else:
            new_item = item
    else:
        new_item = item.clone(shared_data=False, cdims=constant_keys)
        for k, v in item.items():
            new_item[k] = self._add_dimensions(v, dims[::-1], constant_keys)
    if isinstance(new_item, Layout):
        new_item.fixed = True
    return new_item