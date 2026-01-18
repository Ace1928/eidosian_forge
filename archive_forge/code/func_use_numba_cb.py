from __future__ import annotations
import os
from typing import Callable
import pandas._config.config as cf
from pandas._config.config import (
def use_numba_cb(key) -> None:
    from pandas.core.util import numba_
    numba_.set_use_numba(cf.get_option(key))