import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
def _cross_product(self, tuple_key, cache, data_slice):
    """
        Returns a new DynamicMap if the key (tuple form) expresses a
        cross product, otherwise returns None. The cache argument is a
        dictionary (key:element pairs) of all the data found in the
        cache for this key.

        Each key inside the cross product is looked up in the cache
        (self.data) to check if the appropriate element is
        available. Otherwise the element is computed accordingly.

        The data_slice may specify slices into each value in the
        the cross-product.
        """
    if not any((isinstance(el, (list, set)) for el in tuple_key)):
        return None
    if len(tuple_key) == 1:
        product = tuple_key[0]
    else:
        args = [set(el) if isinstance(el, (list, set)) else {el} for el in tuple_key]
        product = itertools.product(*args)
    data = []
    for inner_key in product:
        key = util.wrap_tuple(inner_key)
        if key in cache:
            val = cache[key]
        else:
            val = self._execute_callback(*key)
        if data_slice:
            val = self._dataslice(val, data_slice)
        data.append((key, val))
    product = self.clone(data)
    if data_slice:
        from ..util import Dynamic
        dmap = Dynamic(self, operation=lambda obj, **dynkwargs: obj[data_slice], streams=self.streams)
        dmap.data = product.data
        return dmap
    return product