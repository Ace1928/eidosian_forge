from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Callable, Optional, Union, cast
import numpy as np
from pandas import DataFrame
from seaborn._core.groupby import GroupBy
from seaborn._core.scales import Scale
from seaborn._core.typing import Default
def widths_to_offsets(w):
    return w.shift(1).fillna(0).cumsum() + (w - w.sum()) / 2