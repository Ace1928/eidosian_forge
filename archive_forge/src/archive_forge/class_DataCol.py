from __future__ import annotations
from contextlib import suppress
import copy
from datetime import (
import itertools
import os
import re
from textwrap import dedent
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.lib import is_string_array
from pandas._libs.tslibs import timezones
from pandas.compat._optional import import_optional_dependency
from pandas.compat.pickle_compat import patch_pickle
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import array_equivalent
from pandas import (
from pandas.core.arrays import (
import pandas.core.common as com
from pandas.core.computation.pytables import (
from pandas.core.construction import extract_array
from pandas.core.indexes.api import ensure_index
from pandas.core.internals import (
from pandas.io.common import stringify_path
from pandas.io.formats.printing import (
class DataCol(IndexCol):
    """
    a data holding column, by definition this is not indexable

    Parameters
    ----------
    data   : the actual data
    cname  : the column name in the table to hold the data (typically
                values)
    meta   : a string description of the metadata
    metadata : the actual metadata
    """
    is_an_indexable = False
    is_data_indexable = False
    _info_fields = ['tz', 'ordered']

    def __init__(self, name: str, values=None, kind=None, typ=None, cname: str | None=None, pos=None, tz=None, ordered=None, table=None, meta=None, metadata=None, dtype: DtypeArg | None=None, data=None) -> None:
        super().__init__(name=name, values=values, kind=kind, typ=typ, pos=pos, cname=cname, tz=tz, ordered=ordered, table=table, meta=meta, metadata=metadata)
        self.dtype = dtype
        self.data = data

    @property
    def dtype_attr(self) -> str:
        return f'{self.name}_dtype'

    @property
    def meta_attr(self) -> str:
        return f'{self.name}_meta'

    def __repr__(self) -> str:
        temp = tuple(map(pprint_thing, (self.name, self.cname, self.dtype, self.kind, self.shape)))
        return ','.join([f'{key}->{value}' for key, value in zip(['name', 'cname', 'dtype', 'kind', 'shape'], temp)])

    def __eq__(self, other: object) -> bool:
        """compare 2 col items"""
        return all((getattr(self, a, None) == getattr(other, a, None) for a in ['name', 'cname', 'dtype', 'pos']))

    def set_data(self, data: ArrayLike) -> None:
        assert data is not None
        assert self.dtype is None
        data, dtype_name = _get_data_and_dtype_name(data)
        self.data = data
        self.dtype = dtype_name
        self.kind = _dtype_to_kind(dtype_name)

    def take_data(self):
        """return the data"""
        return self.data

    @classmethod
    def _get_atom(cls, values: ArrayLike) -> Col:
        """
        Get an appropriately typed and shaped pytables.Col object for values.
        """
        dtype = values.dtype
        itemsize = dtype.itemsize
        shape = values.shape
        if values.ndim == 1:
            shape = (1, values.size)
        if isinstance(values, Categorical):
            codes = values.codes
            atom = cls.get_atom_data(shape, kind=codes.dtype.name)
        elif lib.is_np_dtype(dtype, 'M') or isinstance(dtype, DatetimeTZDtype):
            atom = cls.get_atom_datetime64(shape)
        elif lib.is_np_dtype(dtype, 'm'):
            atom = cls.get_atom_timedelta64(shape)
        elif is_complex_dtype(dtype):
            atom = _tables().ComplexCol(itemsize=itemsize, shape=shape[0])
        elif is_string_dtype(dtype):
            atom = cls.get_atom_string(shape, itemsize)
        else:
            atom = cls.get_atom_data(shape, kind=dtype.name)
        return atom

    @classmethod
    def get_atom_string(cls, shape, itemsize):
        return _tables().StringCol(itemsize=itemsize, shape=shape[0])

    @classmethod
    def get_atom_coltype(cls, kind: str) -> type[Col]:
        """return the PyTables column class for this column"""
        if kind.startswith('uint'):
            k4 = kind[4:]
            col_name = f'UInt{k4}Col'
        elif kind.startswith('period'):
            col_name = 'Int64Col'
        else:
            kcap = kind.capitalize()
            col_name = f'{kcap}Col'
        return getattr(_tables(), col_name)

    @classmethod
    def get_atom_data(cls, shape, kind: str) -> Col:
        return cls.get_atom_coltype(kind=kind)(shape=shape[0])

    @classmethod
    def get_atom_datetime64(cls, shape):
        return _tables().Int64Col(shape=shape[0])

    @classmethod
    def get_atom_timedelta64(cls, shape):
        return _tables().Int64Col(shape=shape[0])

    @property
    def shape(self):
        return getattr(self.data, 'shape', None)

    @property
    def cvalues(self):
        """return my cython values"""
        return self.data

    def validate_attr(self, append) -> None:
        """validate that we have the same order as the existing & same dtype"""
        if append:
            existing_fields = getattr(self.attrs, self.kind_attr, None)
            if existing_fields is not None and existing_fields != list(self.values):
                raise ValueError('appended items do not match existing items in table!')
            existing_dtype = getattr(self.attrs, self.dtype_attr, None)
            if existing_dtype is not None and existing_dtype != self.dtype:
                raise ValueError('appended items dtype do not match existing items dtype in table!')

    def convert(self, values: np.ndarray, nan_rep, encoding: str, errors: str):
        """
        Convert the data from this selection to the appropriate pandas type.

        Parameters
        ----------
        values : np.ndarray
        nan_rep :
        encoding : str
        errors : str

        Returns
        -------
        index : listlike to become an Index
        data : ndarraylike to become a column
        """
        assert isinstance(values, np.ndarray), type(values)
        if values.dtype.fields is not None:
            values = values[self.cname]
        assert self.typ is not None
        if self.dtype is None:
            converted, dtype_name = _get_data_and_dtype_name(values)
            kind = _dtype_to_kind(dtype_name)
        else:
            converted = values
            dtype_name = self.dtype
            kind = self.kind
        assert isinstance(converted, np.ndarray)
        meta = _ensure_decoded(self.meta)
        metadata = self.metadata
        ordered = self.ordered
        tz = self.tz
        assert dtype_name is not None
        dtype = _ensure_decoded(dtype_name)
        if dtype.startswith('datetime64'):
            converted = _set_tz(converted, tz, coerce=True)
        elif dtype == 'timedelta64':
            converted = np.asarray(converted, dtype='m8[ns]')
        elif dtype == 'date':
            try:
                converted = np.asarray([date.fromordinal(v) for v in converted], dtype=object)
            except ValueError:
                converted = np.asarray([date.fromtimestamp(v) for v in converted], dtype=object)
        elif meta == 'category':
            categories = metadata
            codes = converted.ravel()
            if categories is None:
                categories = Index([], dtype=np.float64)
            else:
                mask = isna(categories)
                if mask.any():
                    categories = categories[~mask]
                    codes[codes != -1] -= mask.astype(int).cumsum()._values
            converted = Categorical.from_codes(codes, categories=categories, ordered=ordered, validate=False)
        else:
            try:
                converted = converted.astype(dtype, copy=False)
            except TypeError:
                converted = converted.astype('O', copy=False)
        if _ensure_decoded(kind) == 'string':
            converted = _unconvert_string_array(converted, nan_rep=nan_rep, encoding=encoding, errors=errors)
        return (self.values, converted)

    def set_attr(self) -> None:
        """set the data for this column"""
        setattr(self.attrs, self.kind_attr, self.values)
        setattr(self.attrs, self.meta_attr, self.meta)
        assert self.dtype is not None
        setattr(self.attrs, self.dtype_attr, self.dtype)