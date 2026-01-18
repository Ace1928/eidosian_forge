from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def with_spec(spec: DatasetSpec, seed: int | None=None):
    """Generate a random dataset according to provided spec

    Parameters
    ----------
    spec : DatasetSpec
        Specify all the parameters of the dataset
    seed: int (optional)
        Randomstate seed

    Notes
    -----
    This API is still experimental, and will likely change in the future

    Examples
    --------
    >>> from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, with_spec
    >>> ddf = with_spec(
    ...     DatasetSpec(
    ...         npartitions=10,
    ...         nrecords=10_000,
    ...         column_specs=[
    ...             ColumnSpec(dtype=int, number=2, prefix="p"),
    ...             ColumnSpec(dtype=int, number=2, prefix="n", method="normal"),
    ...             ColumnSpec(dtype=float, number=2, prefix="f"),
    ...             ColumnSpec(dtype=str, prefix="s", number=2, random=True, length=10),
    ...             ColumnSpec(dtype="category", prefix="c", choices=["Y", "N"]),
    ...         ],
    ...     ), seed=42)
    >>> ddf.head(10)  # doctest: +SKIP
         p1    p2    n1    n2        f1        f2          s1          s2 c1
    0  1002   972  -811    20  0.640846 -0.176875  L#h98#}J`?  _8C607/:6e  N
    1   985   982 -1663  -777  0.790257  0.792796  u:XI3,omoZ  w~@ /d)'-@  N
    2   947   970   799  -269  0.740869 -0.118413  O$dnwCuq\\  !WtSe+(;#9  Y
    3  1003   983  1133   521 -0.987459  0.278154  j+Qr_2{XG&  &XV7cy$y1T  Y
    4  1017  1049   826     5 -0.875667 -0.744359  \x04bJ3E-{:o  {+jC).?vK+  Y
    5   984  1017  -492  -399  0.748181  0.293761  ~zUNHNgD"!  yuEkXeVot|  Y
    6   992  1027  -856    67 -0.125132 -0.234529  j.7z;o]Gc9  g|Fi5*}Y92  Y
    7  1011   974   762 -1223  0.471696  0.937935  yT?j~N/-u]  JhEB[W-}^$  N
    8   984   974   856    74  0.109963  0.367864  _j"&@ i&;/  OYXQ)w{hoH  N
    9  1030  1001  -792  -262  0.435587 -0.647970  Pmrwl{{|.K  3UTqM$86Sg  N
    """
    if len(spec.column_specs) == 0:
        spec.column_specs = [ColumnSpec(prefix='i', dtype='int64', low=0, high=1000000, random=True), ColumnSpec(prefix='f', dtype=float, random=True), ColumnSpec(prefix='c', dtype='category', choices=['a', 'b', 'c', 'd']), ColumnSpec(prefix='s', dtype=str)]
    columns = []
    dtypes = {}
    partition_freq: str | int | None
    step: str | int
    if isinstance(spec.index_spec, DatetimeIndexSpec):
        start = pd.Timestamp(spec.index_spec.start)
        step = spec.index_spec.freq
        partition_freq = spec.index_spec.partition_freq
        end = pd.Timestamp(spec.index_spec.start) + spec.nrecords * pd.Timedelta(step)
        divisions = list(pd.date_range(start=start, end=end, freq=partition_freq))
        if divisions[-1] < end:
            divisions.append(end)
        meta_start, meta_end = (start, start + pd.Timedelta(step))
    elif isinstance(spec.index_spec, RangeIndexSpec):
        step = spec.index_spec.step
        partition_freq = spec.nrecords * step // spec.npartitions
        end = spec.nrecords * step - 1
        divisions = list(pd.RangeIndex(0, stop=end, step=partition_freq))
        if divisions[-1] < end + 1:
            divisions.append(end + 1)
        meta_start, meta_end = (0, step)
    else:
        raise ValueError(f'Unhandled index: {spec.index_spec}')
    kwargs: dict[str, Any] = {'freq': step}
    for col in spec.column_specs:
        if col.prefix:
            prefix = col.prefix
        elif isinstance(col.dtype, str):
            prefix = re.sub('[^a-zA-Z0-9]', '_', f'{col.dtype}').rstrip('_')
        elif hasattr(col.dtype, 'name'):
            prefix = col.dtype.name
        else:
            prefix = col.dtype.__name__
        for i in range(col.number):
            col_n = i + 1
            while (col_name := f'{prefix}{col_n}') in dtypes:
                col_n = col_n + 1
            columns.append(col_name)
            dtypes[col_name] = col.dtype
            kwargs.update({f'{col_name}_{k}': v for k, v in asdict(col).items() if k not in {'prefix', 'number', 'kwargs'} and v not in (None, [])})
            for kw_name, kw_val in col.kwargs.items():
                kwargs[f'{col_name}_{kw_name}'] = kw_val
    npartitions = len(divisions) - 1
    if seed is None:
        state_data = cast(list[Any], np.random.randint(int(2000000000.0), size=npartitions))
    else:
        state_data = random_state_data(npartitions, seed)
    parts = [(divisions[i:i + 2], state_data[i]) for i in range(npartitions)]
    from dask.dataframe import _dask_expr_enabled
    if _dask_expr_enabled():
        from dask_expr import from_map
        k = {}
    else:
        from dask.dataframe.io.io import from_map
        k = {'token': tokenize(0, spec.nrecords, dtypes, step, partition_freq, state_data)}
    return from_map(MakeDataframePart(spec.index_spec.dtype, dtypes, kwargs, columns=columns), parts, meta=make_dataframe_part(spec.index_spec.dtype, meta_start, meta_end, dtypes, columns, state_data[0], kwargs), divisions=divisions, label='make-random', enforce_metadata=False, **k)