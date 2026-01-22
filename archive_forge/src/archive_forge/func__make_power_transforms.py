from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
def _make_power_transforms(exp: float) -> TransFuncs:

    def forward(x):
        return np.sign(x) * np.power(np.abs(x), exp)

    def inverse(x):
        return np.sign(x) * np.power(np.abs(x), 1 / exp)
    return (forward, inverse)