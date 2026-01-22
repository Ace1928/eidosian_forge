from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
class RaggedArray(ExtensionArray):
    """
    Pandas ExtensionArray to represent ragged arrays

    Methods not otherwise documented here are inherited from ExtensionArray;
    please see the corresponding method on that class for the docstring
    """

    def __init__(self, data, dtype=None, copy=False):
        """
        Construct a RaggedArray

        Parameters
        ----------
        data: list or array or dict or RaggedArray
            * list or 1D-array: A List or 1D array of lists or 1D arrays that
                                should be represented by the RaggedArray

            * dict: A dict containing 'start_indices' and 'flat_array' keys
                    with numpy array values where:
                    - flat_array:  numpy array containing concatenation
                                   of all nested arrays to be represented
                                   by this ragged array
                    - start_indices: unsigned integer numpy array the same
                                     length as the ragged array where values
                                     represent the index into flat_array where
                                     the corresponding ragged array element
                                     begins
            * RaggedArray: A RaggedArray instance to copy

        dtype: RaggedDtype or np.dtype or str or None (default None)
            Datatype to use to store underlying values from data.
            If none (the default) then dtype will be determined using the
            numpy.result_type function.
        copy : bool (default False)
            Whether to deep copy the input arrays. Only relevant when `data`
            has type `dict` or `RaggedArray`. When data is a `list` or
            `array`, input arrays are always copied.
        """
        if isinstance(data, dict) and all((k in data for k in ['start_indices', 'flat_array'])):
            _validate_ragged_properties(start_indices=data['start_indices'], flat_array=data['flat_array'])
            self._start_indices = data['start_indices']
            self._flat_array = data['flat_array']
            dtype = self._flat_array.dtype
            if copy:
                self._start_indices = self._start_indices.copy()
                self._flat_array = self._flat_array.copy()
        elif isinstance(data, RaggedArray):
            self._flat_array = data.flat_array
            self._start_indices = data.start_indices
            dtype = self._flat_array.dtype
            if copy:
                self._start_indices = self._start_indices.copy()
                self._flat_array = self._flat_array.copy()
        else:
            index_len = len(data)
            buffer_len = sum((len(datum) if not missing(datum) else 0 for datum in data))
            for nbits in [8, 16, 32, 64]:
                start_indices_dtype = 'uint' + str(nbits)
                max_supported = np.iinfo(start_indices_dtype).max
                if buffer_len <= max_supported:
                    break
            if dtype is None:
                non_missing = [np.atleast_1d(v) for v in data if not missing(v)]
                if non_missing:
                    dtype = np.result_type(*non_missing)
                else:
                    dtype = 'float64'
            elif isinstance(dtype, RaggedDtype):
                dtype = dtype.subtype
            self._start_indices = np.zeros(index_len, dtype=start_indices_dtype)
            self._flat_array = np.zeros(buffer_len, dtype=dtype)
            next_start_ind = 0
            for i, array_el in enumerate(data):
                n = len(array_el) if not missing(array_el) else 0
                self._start_indices[i] = next_start_ind
                if not n:
                    continue
                self._flat_array[next_start_ind:next_start_ind + n] = array_el
                next_start_ind += n
        self._dtype = RaggedDtype(dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, RaggedArray):
            if len(other) != len(self):
                raise ValueError('\nCannot check equality of RaggedArray values of unequal length\n    len(ra1) == {len_ra1}\n    len(ra2) == {len_ra2}'.format(len_ra1=len(self), len_ra2=len(other)))
            result = _eq_ragged_ragged(self.start_indices, self.flat_array, other.start_indices, other.flat_array)
        else:
            if not isinstance(other, np.ndarray):
                other_array = np.asarray(other)
            else:
                other_array = other
            if other_array.ndim == 1 and other_array.dtype.kind != 'O':
                result = _eq_ragged_scalar(self.start_indices, self.flat_array, other_array)
            elif other_array.ndim == 1 and other_array.dtype.kind == 'O' and (len(other_array) == len(self)):
                result = _eq_ragged_ndarray1d(self.start_indices, self.flat_array, other_array)
            elif other_array.ndim == 2 and other_array.dtype.kind != 'O' and (other_array.shape[0] == len(self)):
                result = _eq_ragged_ndarray2d(self.start_indices, self.flat_array, other_array)
            else:
                raise ValueError('\nCannot check equality of RaggedArray of length {ra_len} with:\n    {other}'.format(ra_len=len(self), other=repr(other)))
        return result

    def __ne__(self, other):
        return np.logical_not(self == other)

    @property
    def flat_array(self):
        """
        numpy array containing concatenation of all nested arrays

        Returns
        -------
        np.ndarray
        """
        return self._flat_array

    @property
    def start_indices(self):
        """
        unsigned integer numpy array the same length as the ragged array where
        values represent the index into flat_array where the corresponding
        ragged array element begins

        Returns
        -------
        np.ndarray
        """
        return self._start_indices

    def __len__(self):
        return len(self._start_indices)

    def __getitem__(self, item):
        err_msg = 'Only integers, slices and integer or booleanarrays are valid indices.'
        if isinstance(item, Integral):
            if item < -len(self) or item >= len(self):
                raise IndexError('{item} is out of bounds'.format(item=item))
            else:
                if item < 0:
                    item += len(self)
                slice_start = self.start_indices[item]
                slice_end = self.start_indices[item + 1] if item + 1 <= len(self) - 1 else len(self.flat_array)
                return self.flat_array[slice_start:slice_end] if slice_end != slice_start else np.nan
        elif type(item) == slice:
            data = []
            selected_indices = np.arange(len(self))[item]
            for selected_index in selected_indices:
                data.append(self[selected_index])
            return RaggedArray(data, dtype=self.flat_array.dtype)
        elif isinstance(item, (np.ndarray, ExtensionArray, list, tuple)):
            if isinstance(item, (np.ndarray, ExtensionArray)):
                kind = item.dtype.kind
            else:
                item = pd.array(item)
                kind = item.dtype.kind
            if len(item) == 0:
                return self.take([], allow_fill=False)
            elif kind == 'b':
                if len(item) != len(self):
                    raise IndexError('Boolean index has wrong length: {} instead of {}'.format(len(item), len(self)))
                isna = pd.isna(item)
                if isna.any():
                    if Version(pd.__version__) > Version('1.0.1'):
                        item[isna] = False
                    else:
                        raise ValueError('Cannot mask with a boolean indexer containing NA values')
                data = []
                for i, m in enumerate(item):
                    if m:
                        data.append(self[i])
                return RaggedArray(data, dtype=self.flat_array.dtype)
            elif kind in ('i', 'u'):
                if any(pd.isna(item)):
                    raise ValueError('Cannot index with an integer indexer containing NA values')
                return self.take(item, allow_fill=False)
            else:
                raise IndexError(err_msg)
        else:
            raise IndexError(err_msg)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return RaggedArray(scalars, dtype=dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return RaggedArray([_RaggedElement.array_or_nan(v) for v in values], dtype=original.flat_array.dtype)

    def _as_ragged_element_array(self):
        return np.array([_RaggedElement.ragged_or_nan(self[i]) for i in range(len(self))])

    def _values_for_factorize(self):
        return (self._as_ragged_element_array(), np.nan)

    def _values_for_argsort(self):
        return self._as_ragged_element_array()

    def unique(self):
        from pandas import unique
        uniques = unique(self._as_ragged_element_array())
        return self._from_sequence([_RaggedElement.array_or_nan(v) for v in uniques], dtype=self.dtype)

    def fillna(self, value=None, method=None, limit=None):
        from pandas.util._validators import validate_fillna_kwargs
        from pandas.core.missing import get_fill_func
        value, method = validate_fillna_kwargs(value, method)
        mask = self.isna()
        if isinstance(value, RaggedArray):
            if len(value) != len(self):
                raise ValueError("Length of 'value' does not match. Got ({})  expected {}".format(len(value), len(self)))
            value = value[mask]
        if mask.any():
            if method is not None:
                func = get_fill_func(method)
                new_values = func(self.astype(object), limit=limit, mask=mask)
                new_values = self._from_sequence(new_values, dtype=self.dtype)
            else:
                new_values = list(self)
                mask_indices, = np.where(mask)
                for ind in mask_indices:
                    new_values[ind] = value
                new_values = self._from_sequence(new_values, dtype=self.dtype)
        else:
            new_values = self.copy()
        return new_values

    def shift(self, periods=1, fill_value=None):
        if not len(self) or periods == 0:
            return self.copy()
        if fill_value is None:
            fill_value = np.nan
        empty = self._from_sequence([fill_value] * min(abs(periods), len(self)), dtype=self.dtype)
        if periods > 0:
            a = empty
            b = self[:-periods]
        else:
            a = self[abs(periods):]
            b = empty
        return self._concat_same_type([a, b])

    def searchsorted(self, value, side='left', sorter=None):
        arr = self._as_ragged_element_array()
        if isinstance(value, RaggedArray):
            search_value = value._as_ragged_element_array()
        else:
            search_value = _RaggedElement(value)
        return arr.searchsorted(search_value, side=side, sorter=sorter)

    def isna(self):
        stop_indices = np.hstack([self.start_indices[1:], [len(self.flat_array)]])
        element_lengths = stop_indices - self.start_indices
        return element_lengths == 0

    def take(self, indices, allow_fill=False, fill_value=None):
        if allow_fill:
            invalid_inds = [i for i in indices if i < -1]
            if invalid_inds:
                raise ValueError('\nInvalid indices for take with allow_fill True: {inds}'.format(inds=invalid_inds[:9]))
            sequence = [self[i] if i >= 0 else fill_value for i in indices]
        else:
            if len(self) == 0 and len(indices) > 0:
                raise IndexError('cannot do a non-empty take from an empty axis|out of bounds')
            sequence = [self[i] for i in indices]
        return RaggedArray(sequence, dtype=self.flat_array.dtype)

    def copy(self, deep=False):
        data = dict(flat_array=self.flat_array, start_indices=self.start_indices)
        return RaggedArray(data, copy=deep)

    @classmethod
    def _concat_same_type(cls, to_concat):
        flat_array = np.hstack([ra.flat_array for ra in to_concat])
        offsets = np.hstack([[0], np.cumsum([len(ra.flat_array) for ra in to_concat[:-1]])])
        start_indices = np.hstack([ra.start_indices + offset for offset, ra in zip(offsets, to_concat)])
        return RaggedArray(dict(flat_array=flat_array, start_indices=start_indices), copy=False)

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._flat_array.nbytes + self._start_indices.nbytes

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, RaggedDtype):
            if copy:
                return self.copy()
            return self
        elif is_extension_array_dtype(dtype):
            return dtype.construct_array_type()._from_sequence(np.asarray(self))
        return np.array([v for v in self], dtype=dtype, copy=copy)

    def tolist(self):
        if self.ndim > 1:
            return [item.tolist() for item in self]
        else:
            return list(self)

    def __array__(self, dtype=None):
        dtype = np.dtype(object) if dtype is None else np.dtype(dtype)
        return np.asarray(self.tolist(), dtype=dtype)

    def duplicated(self, *args, **kwargs):
        msg = 'duplicated is not implemented for RaggedArray'
        raise NotImplementedError(msg)