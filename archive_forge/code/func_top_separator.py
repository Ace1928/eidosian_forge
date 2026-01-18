from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def top_separator(self) -> str:
    return '\\toprule'