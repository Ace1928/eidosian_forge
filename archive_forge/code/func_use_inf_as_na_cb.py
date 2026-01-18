from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def use_inf_as_na_cb(key) -> None:
    from pandas.core.dtypes.missing import _use_inf_as_na
    _use_inf_as_na(key)