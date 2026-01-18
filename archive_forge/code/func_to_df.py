from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
def to_df(self) -> pd.DataFrame:
    """ Convert this data source to pandas ``DataFrame``.

        Returns:
            DataFrame

        """
    import pandas as pd
    return pd.DataFrame(self.data)