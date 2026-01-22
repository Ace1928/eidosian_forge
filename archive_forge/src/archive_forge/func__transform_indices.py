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
def _transform_indices(self, key):
    """Snaps indices into the GridSpace to the closest coordinate.

        Args:
            key: Tuple index into the GridSpace

        Returns:
            Transformed key snapped to closest numeric coordinates
        """
    ndims = self.ndims
    if all((not (isinstance(el, slice) or callable(el)) for el in key)):
        dim_inds = []
        for dim in self.kdims:
            dim_type = self.get_dimension_type(dim)
            if isinstance(dim_type, type) and issubclass(dim_type, Number):
                dim_inds.append(self.get_dimension_index(dim))
        str_keys = iter((key[i] for i in range(self.ndims) if i not in dim_inds))
        num_keys = []
        if len(dim_inds):
            keys = list({tuple((k[i] if ndims > 1 else k for i in dim_inds)) for k in self.keys()})
            q = np.array([tuple((key[i] if ndims > 1 else key for i in dim_inds))])
            idx = np.argmin([np.inner(q - np.array(x), q - np.array(x)) if len(dim_inds) == 2 else np.abs(q - x) for x in keys])
            num_keys = iter(keys[idx])
        key = tuple((next(num_keys) if i in dim_inds else next(str_keys) for i in range(self.ndims)))
    elif any((not (isinstance(el, slice) or callable(el)) for el in key)):
        keys = self.keys()
        for i, k in enumerate(key):
            if isinstance(k, slice):
                continue
            dim_keys = np.array([ke[i] for ke in keys])
            if dim_keys.dtype.kind in 'OSU':
                continue
            snapped_val = dim_keys[np.argmin(np.abs(dim_keys - k))]
            key = list(key)
            key[i] = snapped_val
        key = tuple(key)
    return key