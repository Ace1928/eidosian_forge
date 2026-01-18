from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
def to_xarray(self):
    """
        Return an xarray object from the pandas object.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Data in the pandas structure converted to Dataset if the object is
            a DataFrame, or a DataArray if the object is a Series.

        See Also
        --------
        DataFrame.to_hdf : Write DataFrame to an HDF5 file.
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

        Notes
        -----
        See the `xarray docs <https://xarray.pydata.org/en/stable/>`__

        Examples
        --------
        >>> df = pd.DataFrame([('falcon', 'bird', 389.0, 2),
        ...                    ('parrot', 'bird', 24.0, 2),
        ...                    ('lion', 'mammal', 80.5, 4),
        ...                    ('monkey', 'mammal', np.nan, 4)],
        ...                   columns=['name', 'class', 'max_speed',
        ...                            'num_legs'])
        >>> df
             name   class  max_speed  num_legs
        0  falcon    bird      389.0         2
        1  parrot    bird       24.0         2
        2    lion  mammal       80.5         4
        3  monkey  mammal        NaN         4

        >>> df.to_xarray()  # doctest: +SKIP
        <xarray.Dataset>
        Dimensions:    (index: 4)
        Coordinates:
          * index      (index) int64 32B 0 1 2 3
        Data variables:
            name       (index) object 32B 'falcon' 'parrot' 'lion' 'monkey'
            class      (index) object 32B 'bird' 'bird' 'mammal' 'mammal'
            max_speed  (index) float64 32B 389.0 24.0 80.5 nan
            num_legs   (index) int64 32B 2 2 4 4

        >>> df['max_speed'].to_xarray()  # doctest: +SKIP
        <xarray.DataArray 'max_speed' (index: 4)>
        array([389. ,  24. ,  80.5,   nan])
        Coordinates:
          * index    (index) int64 0 1 2 3

        >>> dates = pd.to_datetime(['2018-01-01', '2018-01-01',
        ...                         '2018-01-02', '2018-01-02'])
        >>> df_multiindex = pd.DataFrame({'date': dates,
        ...                               'animal': ['falcon', 'parrot',
        ...                                          'falcon', 'parrot'],
        ...                               'speed': [350, 18, 361, 15]})
        >>> df_multiindex = df_multiindex.set_index(['date', 'animal'])

        >>> df_multiindex
                           speed
        date       animal
        2018-01-01 falcon    350
                   parrot     18
        2018-01-02 falcon    361
                   parrot     15

        >>> df_multiindex.to_xarray()  # doctest: +SKIP
        <xarray.Dataset>
        Dimensions:  (date: 2, animal: 2)
        Coordinates:
          * date     (date) datetime64[ns] 2018-01-01 2018-01-02
          * animal   (animal) object 'falcon' 'parrot'
        Data variables:
            speed    (date, animal) int64 350 18 361 15
        """
    xarray = import_optional_dependency('xarray')
    if self.ndim == 1:
        return xarray.DataArray.from_series(self)
    else:
        return xarray.Dataset.from_dataframe(self)