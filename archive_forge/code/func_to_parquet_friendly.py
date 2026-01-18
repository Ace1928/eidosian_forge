from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def to_parquet_friendly(self, df: T, partition_cols: Optional[List[str]]=None) -> T:
    """Parquet doesn't like pd.ArrowDtype(<nested types>), this function
        converts all nested types to object types

        :param df: the input dataframe
        :param partition_cols: the partition columns, if any, default None
        :return: the converted dataframe
        """
    pcols = partition_cols or []
    changed = False
    new_types: Dict[str, Any] = {}
    for k, v in df.dtypes.items():
        if k in pcols:
            new_types[k] = np.dtype(object)
            changed = True
        elif hasattr(pd, 'ArrowDtype') and isinstance(v, pd.ArrowDtype) and pa.types.is_nested(v.pyarrow_dtype):
            new_types[k] = np.dtype(object)
            changed = True
        else:
            new_types[k] = v
    if changed:
        df = df.astype(new_types)
    return df