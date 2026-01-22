from __future__ import annotations
import sys
from typing import (
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypeAlias  # noqa: TCH002
from plotnine import ggplot, guide_colorbar, guide_legend
from plotnine.iapi import strip_label_details
class DataFrameConvertible(Protocol):
    """
    Object that can be converted to a DataFrame
    """

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas dataframe

        Returns
        -------
        :
            Pandas representation of this object.
        """
        ...