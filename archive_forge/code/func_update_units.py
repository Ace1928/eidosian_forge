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
def update_units(self, x):
    """Pass units to the internal converter, potentially updating its mapping."""
    self.converter = mpl.units.registry.get_converter(x)
    if self.converter is not None:
        self.converter.default_units(x, self)
        info = self.converter.axisinfo(self.units, self)
        if info is None:
            return
        if info.majloc is not None:
            self.set_major_locator(info.majloc)
        if info.majfmt is not None:
            self.set_major_formatter(info.majfmt)