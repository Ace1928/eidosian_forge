import os
import numpy as np
import pandas.json as json
from pandas.tslib import iNaT
from pandas.compat import StringIO, long, u
from pandas import compat, isnull
from pandas import Series, DataFrame, to_datetime, MultiIndex
from pandas.io.common import (get_filepath_or_buffer, _get_handle,
from pandas.core.common import AbstractMethodError
from pandas.formats.printing import pprint_thing
from .normalize import _convert_to_line_delimits
from .table_schema import build_table_schema
def _try_convert_data(self, name, data, use_dtypes=True, convert_dates=True):
    """ try to parse a ndarray like into a column by inferring dtype """
    if use_dtypes:
        if self.dtype is False:
            return (data, False)
        elif self.dtype is True:
            pass
        else:
            dtype = self.dtype.get(name) if isinstance(self.dtype, dict) else self.dtype
            if dtype is not None:
                try:
                    dtype = np.dtype(dtype)
                    return (data.astype(dtype), True)
                except Exception:
                    return (data, False)
    if convert_dates:
        new_data, result = self._try_convert_to_date(data)
        if result:
            return (new_data, True)
    result = False
    if data.dtype == 'object':
        try:
            data = data.astype('float64')
            result = True
        except Exception:
            pass
    if data.dtype.kind == 'f':
        if data.dtype != 'float64':
            try:
                data = data.astype('float64')
                result = True
            except Exception:
                pass
    if len(data) and (data.dtype == 'float' or data.dtype == 'object'):
        try:
            new_data = data.astype('int64')
            if (new_data == data).all():
                data = new_data
                result = True
        except Exception:
            pass
    if data.dtype == 'int':
        try:
            data = data.astype('int64')
            result = True
        except Exception:
            pass
    return (data, result)