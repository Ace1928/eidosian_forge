import math
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.dataframe.core import DataFrame
from dask.delayed import delayed
from dask.distributed import Client, get_client
from triad.utils.pandas_like import PD_UTILS, PandasLikeUtils
from triad.utils.pyarrow import to_pandas_dtype
import fugue.api as fa
from fugue.constants import FUGUE_CONF_DEFAULT_PARTITIONS
from ._constants import FUGUE_DASK_CONF_DEFAULT_PARTITIONS
def _add_group_index(df: dd.DataFrame, cols: List[str], shuffle: bool, seed: Any=None) -> Tuple[dd.DataFrame, int]:
    keys = df[cols].drop_duplicates().compute()
    if shuffle:
        keys = keys.sample(frac=1, random_state=seed)
    keys = keys.reset_index(drop=True).assign(**{_FUGUE_DASK_TEMP_IDX_COLUMN: pd.Series(range(len(keys)), dtype=int)})
    df = df.merge(dd.from_pandas(keys, npartitions=1), on=cols, broadcast=True)
    return (df.set_index(_FUGUE_DASK_TEMP_IDX_COLUMN, drop=True), len(keys))