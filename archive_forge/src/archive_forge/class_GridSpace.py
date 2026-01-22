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
class GridSpace(Layoutable, UniformNdMapping):
    """
    Grids are distinct from Layouts as they ensure all contained
    elements to be of the same type. Unlike Layouts, which have
    integer keys, Grids usually have floating point keys, which
    correspond to a grid sampling in some two-dimensional space. This
    two-dimensional space may have to arbitrary dimensions, e.g. for
    2D parameter spaces.
    """
    kdims = param.List(default=[Dimension('X'), Dimension('Y')], bounds=(1, 2))

    def __init__(self, initial_items=None, kdims=None, **params):
        super().__init__(initial_items, kdims=kdims, **params)
        if self.ndims > 2:
            raise Exception('Grids can have no more than two dimensions.')

    def __lshift__(self, other):
        """Adjoins another object to the GridSpace"""
        if isinstance(other, (ViewableElement, UniformNdMapping)):
            return AdjointLayout([self, other])
        elif isinstance(other, AdjointLayout):
            return AdjointLayout(other.data + [self])
        else:
            raise TypeError(f'Cannot append {type(other).__name__} to a AdjointLayout')

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

    def keys(self, full_grid=False):
        """Returns the keys of the GridSpace

        Args:
            full_grid (bool, optional): Return full cross-product of keys

        Returns:
            List of keys
        """
        keys = super().keys()
        if self.ndims == 1 or not full_grid:
            return keys
        dim1_keys = list(dict.fromkeys((k[0] for k in keys)))
        dim2_keys = list(dict.fromkeys((k[1] for k in keys)))
        return [(d1, d2) for d1 in dim1_keys for d2 in dim2_keys]

    @property
    def last(self):
        """
        The last of a GridSpace is another GridSpace
        constituted of the last of the individual elements. To access
        the elements by their X,Y position, either index the position
        directly or use the items() method.
        """
        if self.type == HoloMap:
            last_items = [(k, v.last if isinstance(v, HoloMap) else v) for k, v in self.data.items()]
        else:
            last_items = self.data
        return self.clone(last_items)

    def __len__(self):
        """
        The maximum depth of all the elements. Matches the semantics
        of __len__ used by Maps. For the total number of elements,
        count the full set of keys.
        """
        return max([len(v) if hasattr(v, '__len__') else 1 for v in self.values()] + [0])

    @property
    def shape(self):
        """Returns the 2D shape of the GridSpace as (rows, cols)."""
        keys = self.keys()
        if self.ndims == 1:
            return (len(keys), 1)
        return (len({k[0] for k in keys}), len({k[1] for k in keys}))

    def decollate(self):
        """Packs GridSpace of DynamicMaps into a single DynamicMap that returns a
        GridSpace

        Decollation allows packing a GridSpace of DynamicMaps into a single DynamicMap
        that returns a GridSpace of simple (non-dynamic) elements. All nested streams
        are lifted to the resulting DynamicMap, and are available in the `streams`
        property.  The `callback` property of the resulting DynamicMap is a pure,
        stateless function of the stream values. To avoid stream parameter name
        conflicts, the resulting DynamicMap is configured with
        positional_stream_args=True, and the callback function accepts stream values
        as positional dict arguments.

        Returns:
            DynamicMap that returns a GridSpace
        """
        from .decollate import decollate
        return decollate(self)