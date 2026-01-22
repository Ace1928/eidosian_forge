from types import ModuleType
from typing import Any, Callable, Tuple, Union
import numpy as np
from ray.data.block import AggType, Block, KeyType, T, U
def _lazy_import_pandas() -> LazyModule:
    global _pandas
    if _pandas is None:
        try:
            import pandas as _pandas
        except ModuleNotFoundError:
            _pandas = False
    return _pandas