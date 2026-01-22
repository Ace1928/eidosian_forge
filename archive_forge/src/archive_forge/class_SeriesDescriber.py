from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas._typing import (
from pandas.util._validators import validate_percentile
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.reshape.concat import concat
from pandas.io.formats.format import format_percentiles
class SeriesDescriber(NDFrameDescriberAbstract):
    """Class responsible for creating series description."""
    obj: Series

    def describe(self, percentiles: Sequence[float] | np.ndarray) -> Series:
        describe_func = select_describe_func(self.obj)
        return describe_func(self.obj, percentiles)