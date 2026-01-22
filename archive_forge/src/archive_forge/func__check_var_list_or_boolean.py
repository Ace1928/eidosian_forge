from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._stats.base import Stat
def _check_var_list_or_boolean(self, param: str, grouping_vars: Any) -> None:
    """Do input checks on grouping parameters."""
    value = getattr(self, param)
    if not (isinstance(value, bool) or (isinstance(value, list) and all((isinstance(v, str) for v in value)))):
        param_name = f'{self.__class__.__name__}.{param}'
        raise TypeError(f'{param_name} must be a boolean or list of strings.')
    self._check_grouping_vars(param, grouping_vars, stacklevel=3)