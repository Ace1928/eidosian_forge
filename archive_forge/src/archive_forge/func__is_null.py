from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _is_null(r: Any):
    pd = _lazy_import_pandas()
    if pd:
        return pd.isnull(r)
    try:
        return np.isnan(r)
    except TypeError:
        return r is None