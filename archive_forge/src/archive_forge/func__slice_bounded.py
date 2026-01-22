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
def _slice_bounded(self, tuple_key, data_slice):
    """
        Slices bounded DynamicMaps by setting the soft_ranges on
        key dimensions and applies data slice to cached and dynamic
        values.
        """
    slices = [el for el in tuple_key if isinstance(el, slice)]
    if any((el.step for el in slices)):
        raise Exception('DynamicMap slices cannot have a step argument')
    elif len(slices) not in [0, len(tuple_key)]:
        raise Exception('Slices must be used exclusively or not at all')
    elif not slices:
        return None
    sliced = self.clone(self)
    for i, slc in enumerate(tuple_key):
        start, stop = (slc.start, slc.stop)
        if start is not None and start < sliced.kdims[i].range[0]:
            raise Exception('Requested slice below defined dimension range.')
        if stop is not None and stop > sliced.kdims[i].range[1]:
            raise Exception('Requested slice above defined dimension range.')
        sliced.kdims[i].soft_range = (start, stop)
    if data_slice:
        if not isinstance(sliced, DynamicMap):
            return self._dataslice(sliced, data_slice)
        else:
            from ..util import Dynamic
            if len(self):
                slices = [slice(None) for _ in range(self.ndims)] + list(data_slice)
                sliced = super(DynamicMap, sliced).__getitem__(tuple(slices))
            dmap = Dynamic(self, operation=lambda obj, **dynkwargs: obj[data_slice], streams=self.streams)
            dmap.data = sliced.data
            return dmap
    return sliced