from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
class NdMapping(MultiDimensionalMapping):
    """
    NdMapping supports the same indexing semantics as
    MultiDimensionalMapping but also supports slicing semantics.

    Slicing semantics on an NdMapping is dependent on the ordering
    semantics of the keys. As MultiDimensionalMapping sort the keys, a
    slice on an NdMapping is effectively a way of filtering out the
    keys that are outside the slice range.
    """
    group = param.String(default='NdMapping', constant=True)

    def __getitem__(self, indexslice):
        """
        Allows slicing operations along the key and data
        dimensions. If no data slice is supplied it will return all
        data elements, otherwise it will return the requested slice of
        the data.
        """
        if isinstance(indexslice, np.ndarray) and indexslice.dtype.kind == 'b':
            if not len(indexslice) == len(self):
                raise IndexError('Boolean index must match length of sliced object')
            selection = zip(indexslice, self.data.items())
            return self.clone([item for c, item in selection if c])
        elif isinstance(indexslice, tuple) and indexslice == () and (not self.kdims):
            return self.data[()]
        elif isinstance(indexslice, tuple) and indexslice == () or indexslice is Ellipsis:
            return self
        elif any((Ellipsis is sl for sl in wrap_tuple(indexslice))):
            indexslice = process_ellipses(self, indexslice)
        map_slice, data_slice = self._split_index(indexslice)
        map_slice = self._transform_indices(map_slice)
        map_slice = self._expand_slice(map_slice)
        if all((not (isinstance(el, (slice, set, list, tuple)) or callable(el)) for el in map_slice)):
            return self._dataslice(self.data[map_slice], data_slice)
        else:
            conditions = self._generate_conditions(map_slice)
            items = self.data.items()
            for cidx, (condition, dim) in enumerate(zip(conditions, self.kdims)):
                values = dim.values
                items = [(k, v) for k, v in items if condition(values.index(k[cidx]) if values else k[cidx])]
            sliced_items = []
            for k, v in items:
                val_slice = self._dataslice(v, data_slice)
                if val_slice or isinstance(val_slice, tuple):
                    sliced_items.append((k, val_slice))
            if len(sliced_items) == 0:
                raise KeyError('No items within specified slice.')
            with item_check(False):
                return self.clone(sliced_items)

    def _expand_slice(self, indices):
        """
        Expands slices containing steps into a list.
        """
        keys = list(self.data.keys())
        expanded = []
        for idx, ind in enumerate(indices):
            if isinstance(ind, slice) and ind.step is not None:
                dim_ind = slice(ind.start, ind.stop)
                if dim_ind == slice(None):
                    condition = self._all_condition()
                elif dim_ind.start is None:
                    condition = self._upto_condition(dim_ind)
                elif dim_ind.stop is None:
                    condition = self._from_condition(dim_ind)
                else:
                    condition = self._range_condition(dim_ind)
                dim_vals = unique_iterator((k[idx] for k in keys))
                expanded.append(set([k for k in dim_vals if condition(k)][::int(ind.step)]))
            else:
                expanded.append(ind)
        return tuple(expanded)

    def _transform_indices(self, indices):
        """
        Identity function here but subclasses can implement transforms
        of the dimension indices from one coordinate system to another.
        """
        return indices

    def _generate_conditions(self, map_slice):
        """
        Generates filter conditions used for slicing the data structure.
        """
        conditions = []
        for dim, dim_slice in zip(self.kdims, map_slice):
            if isinstance(dim_slice, slice):
                start, stop = (dim_slice.start, dim_slice.stop)
                if dim.values:
                    values = dim.values
                    dim_slice = slice(None if start is None else values.index(start), None if stop is None else values.index(stop))
                if dim_slice == slice(None):
                    conditions.append(self._all_condition())
                elif start is None:
                    conditions.append(self._upto_condition(dim_slice))
                elif stop is None:
                    conditions.append(self._from_condition(dim_slice))
                else:
                    conditions.append(self._range_condition(dim_slice))
            elif isinstance(dim_slice, (set, list)):
                if dim.values:
                    dim_slice = [dim.values.index(dim_val) for dim_val in dim_slice]
                conditions.append(self._values_condition(dim_slice))
            elif dim_slice is Ellipsis:
                conditions.append(self._all_condition())
            elif callable(dim_slice):
                conditions.append(dim_slice)
            elif isinstance(dim_slice, tuple):
                raise IndexError('Keys may only be selected with sets or lists, not tuples.')
            else:
                if dim.values:
                    dim_slice = dim.values.index(dim_slice)
                conditions.append(self._value_condition(dim_slice))
        return conditions

    def _value_condition(self, value):
        return lambda x: x == value

    def _values_condition(self, values):
        return lambda x: x in values

    def _range_condition(self, slice):
        if slice.step is None:
            lmbd = lambda x: slice.start <= x < slice.stop
        else:
            lmbd = lambda x: slice.start <= x < slice.stop and (not (x - slice.start) % slice.step)
        return lmbd

    def _upto_condition(self, slice):
        if slice.step is None:
            lmbd = lambda x: x < slice.stop
        else:
            lmbd = lambda x: x < slice.stop and (not x % slice.step)
        return lmbd

    def _from_condition(self, slice):
        if slice.step is None:
            lmbd = lambda x: x >= slice.start
        else:
            lmbd = lambda x: x >= slice.start and (x - slice.start) % slice.step
        return lmbd

    def _all_condition(self):
        return lambda x: True