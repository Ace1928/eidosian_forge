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
def _add_rand(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.assign(**{_FUGUE_DASK_TEMP_IDX_COLUMN: pd.Series(dtype=int)})
    if seed is not None:
        np.random.seed(seed)
    return df.assign(**{_FUGUE_DASK_TEMP_IDX_COLUMN: np.random.randint(0, num, len(df))})